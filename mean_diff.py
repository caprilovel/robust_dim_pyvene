import os
from typing import List, Optional, Tuple

import torch
import pyvene as pv
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def build_collector(model: PreTrainedModel, layer: int, component: str = "block_output") -> pv.IntervenableModel:
    """Wrap `model` so that forward calls can collect activations at `layer`."""
    config = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=layer,
                component=component,
                intervention_type=pv.CollectIntervention,
            )
        ]
    )
    return pv.IntervenableModel(config, model)


def collect_activations(
    collector: pv.IntervenableModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    unit_location: int = 0,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """Run `texts` through `collector` and return a [N, hidden_dim] tensor of
    the activations collected at `unit_location` (token index into `base`).

    If `save_path` is given, also save the returned tensor there (as a
    `torch.save`'d `.pt` file, parent directories created as needed) so the
    raw hidden states can be reused without re-running the model, e.g. for
    offline experimentation with different reweighting schemes.
    """
    acts = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            (_, collected), _ = collector(
                base=inputs,
                unit_locations={"base": unit_location},
            )
            acts.append(collected[0].reshape(-1).float())
    acts = torch.stack(acts)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(acts, save_path)

    return acts


def compute_mean_diff(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pos_texts: List[str],
    neg_texts: List[str],
    layer: int,
    component: str = "block_output",
    unit_location: int = 0,
    pos_save_path: Optional[str] = None,
    neg_save_path: Optional[str] = None,
) -> torch.Tensor:
    """Compute the mean-difference steering vector between `pos_texts` and
    `neg_texts` at the given layer/component, using the activation at
    `unit_location`.

    If `pos_save_path`/`neg_save_path` are given, the raw collected
    activations are also cached there (see `collect_activations`).

    Returns a [hidden_dim] tensor: mean(pos_acts) - mean(neg_acts).
    """
    model.eval()
    collector = build_collector(model, layer=layer, component=component)

    pos_acts = collect_activations(collector, tokenizer, pos_texts, unit_location, save_path=pos_save_path)
    neg_acts = collect_activations(collector, tokenizer, neg_texts, unit_location, save_path=neg_save_path)

    return (pos_acts - neg_acts).mean(dim=0)

def compute_geometry_median_diff(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        pos_texts: List[str],
        neg_texts: List[str],
        layer: int,
        component: str = "block_output",
        unit_location: int = 0,
        pos_save_path: Optional[str] = None,
        neg_save_path: Optional[str] = None,
)-> torch.Tensor:
    """Compute the geometric median difference steering vector between `pos_texts` and
    `neg_texts` at the given layer/component, using the activation at
    `unit_location`.

    If `pos_save_path`/`neg_save_path` are given, the raw collected
    activations are also cached there (see `collect_activations`).

    Returns a [hidden_dim] tensor: geometric_median(pos_acts) - geometric_median(neg_acts).
    """
    model.eval()
    collector = build_collector(model, layer=layer, component=component)

    pos_acts = collect_activations(collector, tokenizer, pos_texts, unit_location, save_path=pos_save_path)
    neg_acts = collect_activations(collector, tokenizer, neg_texts, unit_location, save_path=neg_save_path)

    pos_median = pv.geometry_median(pos_acts)
    neg_median = pv.geometry_median(neg_acts)

    return pos_median - neg_median


# --- Iterative reweighted DiM ------------------------------------------------
#
# Plain DiM (`compute_mean_diff` above) is mu_hat = normalize(mean(pos) -
# mean(neg)), which implicitly weights every example equally. If the pool is
# heterogeneous -- some examples point in a very different direction than the
# population mean, e.g. because they come from a different sub-behavior or
# are noisy/mislabeled -- equal weighting lets those examples drag the
# estimate around.
#
# `compute_robust_mean_diff` instead implements a soft, self-consistent
# downweighting scheme, ported from the paired/unpaired estimators in the
# sibling dim_angle_analysis repo (reweighted_dim.py /
# reweighted_dim_unpaired.py), but written for unpaired pos/neg activation
# tensors (the only shape `collect_activations` produces):
#   1. K-fold cross-fitting within each group: fold k's weights are set
#      using the direction estimated from the OTHER folds, so an example
#      never influences its own weight.
#   2. For example x_i in fold k, project onto the current direction mu:
#        a_i     = x_i . mu                    (signed alignment)
#        tau_i^2 = || (I - mu mu^T) x_i ||^2    (out-of-direction "noise")
#      Shrink tau_i^2 toward the fold's mean (single-sample variance
#      estimates are themselves noisy) and form
#        w_i = 1 / (tau_tilde_i^2 + lambda)
#      i.e. upweight examples with small out-of-direction noise; there is
#      no relu/sign-gating here (unlike the paired version) since a
#      negative a_i is meaningless for a single class's own samples around
#      their own mean -- only the noise term discriminates.
#   3. Clip weights to c/N, renormalize, recompute mu_hat from the
#      reweighted group means, repeat T times.

def _fold_assignment(n: int, K: int, generator: torch.Generator) -> torch.Tensor:
    """Randomly assign n items to K folds; returns a length-n LongTensor of fold ids."""
    idx = torch.randperm(n, generator=generator)
    fold_id = torch.empty(n, dtype=torch.long)
    for k, fold in enumerate(idx.chunk(K)):
        fold_id[fold] = k
    return fold_id


def _reweight_one_group(
    X: torch.Tensor,
    mu: torch.Tensor,
    fold_id: torch.Tensor,
    K: int,
    gamma: float,
    lam: float,
    c: float,
) -> torch.Tensor:
    """Return new weights (summing to 1) for one group's samples, given the
    current direction `mu`, using K-fold cross-fitting."""
    n = X.shape[0]
    a = torch.empty(n, dtype=X.dtype)
    tau2 = torch.empty(n, dtype=X.dtype)

    for k in range(K):
        mask = fold_id == k
        Xk = X[mask]
        proj_coef = Xk @ mu
        a[mask] = proj_coef
        resid = Xk - torch.outer(proj_coef, mu)
        tau2[mask] = (resid ** 2).sum(dim=1)

    tau2_tilde = (1 - gamma) * tau2 + gamma * tau2.mean()
    w = 1.0 / (tau2_tilde + lam)
    w = torch.clamp(w, min=0.0, max=c / n)
    return w / w.sum()


def robust_dim(
    pos: torch.Tensor,
    neg: torch.Tensor,
    K: int = 5,
    gamma: float = 0.3,
    lam: float = 0.01,
    c: float = 5.0,
    T: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iteratively reweighted DiM direction for unpaired pos/neg activation
    tensors (see module docstring above for the algorithm).

    Returns (mu, w_pos, w_neg, plain_mu):
      mu       -- unit vector, the final reweighted direction
      w_pos    -- per-example weights for `pos` (sum to 1)
      w_neg    -- per-example weights for `neg` (sum to 1)
      plain_mu -- unit vector, the plain (unweighted) DiM direction, for comparison
    """
    generator = torch.Generator().manual_seed(seed)
    fold_pos = _fold_assignment(pos.shape[0], K, generator)
    fold_neg = _fold_assignment(neg.shape[0], K, generator)

    plain_mu = torch.nn.functional.normalize(pos.mean(dim=0) - neg.mean(dim=0), dim=0)
    mu = plain_mu.clone()

    w_pos = torch.full((pos.shape[0],), 1.0 / pos.shape[0], dtype=pos.dtype)
    w_neg = torch.full((neg.shape[0],), 1.0 / neg.shape[0], dtype=neg.dtype)

    for t in range(T):
        w_pos = _reweight_one_group(pos, mu, fold_pos, K, gamma, lam, c)
        w_neg = _reweight_one_group(neg, mu, fold_neg, K, gamma, lam, c)

        mu_pos = (w_pos[:, None] * pos).sum(dim=0)
        mu_neg = (w_neg[:, None] * neg).sum(dim=0)
        mu = torch.nn.functional.normalize(mu_pos - mu_neg, dim=0)

        if verbose:
            cos = torch.clamp(mu @ plain_mu, -1.0, 1.0)
            angle = torch.rad2deg(torch.arccos(cos)).item()
            n_eff_pos = 1.0 / (w_pos ** 2).sum().item()
            n_eff_neg = 1.0 / (w_neg ** 2).sum().item()
            print(f"  iter {t + 1}/{T}: angle-to-plain={angle:.2f} deg, "
                  f"N_eff_pos={n_eff_pos:.1f}/{pos.shape[0]}, "
                  f"N_eff_neg={n_eff_neg:.1f}/{neg.shape[0]}")

    return mu, w_pos, w_neg, plain_mu


def compute_robust_mean_diff(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pos_texts: List[str],
    neg_texts: List[str],
    layer: int,
    component: str = "block_output",
    unit_location: int = 0,
    pos_save_path: Optional[str] = None,
    neg_save_path: Optional[str] = None,
    K: int = 5,
    gamma: float = 0.3,
    lam: float = 0.01,
    c: float = 5.0,
    T: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect activations for `pos_texts`/`neg_texts` and compute the
    iteratively reweighted DiM direction between them (see `robust_dim`).

    If `pos_save_path`/`neg_save_path` are given, the raw collected
    activations are also cached there (see `collect_activations`).

    Returns (mu, w_pos, w_neg, plain_mu) -- see `robust_dim`.
    """
    model.eval()
    collector = build_collector(model, layer=layer, component=component)

    pos_acts = collect_activations(collector, tokenizer, pos_texts, unit_location, save_path=pos_save_path)
    neg_acts = collect_activations(collector, tokenizer, neg_texts, unit_location, save_path=neg_save_path)

    return robust_dim(pos_acts, neg_acts, K=K, gamma=gamma, lam=lam, c=c, T=T, seed=seed, verbose=verbose)
