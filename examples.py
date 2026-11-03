# Runnable demos for the direction estimators in mean_diff.py: plain DiM,
# geometric-median DiM, and iterative reweighted DiM. Run as
# `python examples.py {mean_diff,geometry_median,robust}`.

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import RefusalDatasetFactory
from mean_diff import compute_geometry_median_diff, compute_mean_diff, compute_robust_mean_diff

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
LAYER = 15


def _load_model_and_data():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    factory = RefusalDatasetFactory()
    pos_instructions = factory.get("train", "pos")
    neg_instructions = factory.get("train", "neg")
    return model, tokenizer, pos_instructions, neg_instructions


def run_mean_diff():
    model, tokenizer, pos, neg = _load_model_and_data()
    steering_vector = compute_mean_diff(model, tokenizer, pos, neg, layer=LAYER)
    print("steering vector shape:", steering_vector.shape)


def run_geometry_median():
    model, tokenizer, pos, neg = _load_model_and_data()
    steering_vector = compute_geometry_median_diff(model, tokenizer, pos, neg, layer=LAYER)
    print("steering vector shape:", steering_vector.shape)


def run_robust():
    model, tokenizer, pos, neg = _load_model_and_data()
    mu, w_pos, w_neg, plain_mu = compute_robust_mean_diff(model, tokenizer, pos, neg, layer=LAYER, verbose=True)

    cos = torch.clamp(mu @ plain_mu, -1.0, 1.0)
    angle = torch.rad2deg(torch.arccos(cos)).item()
    print(f"\nFinal angle between reweighted mu and plain DiM: {angle:.2f} deg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", choices=["mean_diff", "geometry_median", "robust"])
    args = parser.parse_args()

    {
        "mean_diff": run_mean_diff,
        "geometry_median": run_geometry_median,
        "robust": run_robust,
    }[args.demo]()


if __name__ == "__main__":
    main()
