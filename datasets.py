import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from omegaconf import OmegaConf

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")
DATASETS_CONFIG_PATH = os.path.join(CONFIG_DIR, "datasets.yaml")


def _load_dataset_config(name: str) -> dict:
    """Load the config section for a dataset factory by name (e.g. "Refusal")."""
    cfg = OmegaConf.load(DATASETS_CONFIG_PATH)
    section = cfg.get(name, {})
    return OmegaConf.to_container(section, resolve=True)


@dataclass
class DatasetSplit:
    pos: List[str] = field(default_factory=list)
    neg: List[str] = field(default_factory=list)
    neutral: List[str] = field(default_factory=list)


class DatasetFactory(ABC):
    """Base factory for datasets split into pos/neg/neutral examples.

    Subclasses implement `_load_raw` to return the raw pos/neg/neutral
    examples (as strings), and this class handles the train/test split.
    """

    def __init__(
        self,
        train_frac: float = 0.8,
        seed: int = 0,
        flip_frac: float = 0.0,
        flip_test: bool = False,
    ):
        assert 0.0 < train_frac <= 1.0
        assert 0.0 <= flip_frac <= 1.0
        self.train_frac = train_frac
        self.seed = seed
        self.flip_frac = flip_frac
        self.flip_test = flip_test

        self._train: Optional[DatasetSplit] = None
        self._test: Optional[DatasetSplit] = None

    @abstractmethod
    def _load_raw(self) -> DatasetSplit:
        """Return the full (unsplit) pos/neg/neutral examples."""
        raise NotImplementedError

    def _split_list(self, items: List[str]) -> Tuple[List[str], List[str]]:
        import random

        items = list(items)
        random.Random(self.seed).shuffle(items)
        n_train = int(len(items) * self.train_frac)
        return items[:n_train], items[n_train:]

    def _flip_labels(self, split: DatasetSplit, seed: int) -> DatasetSplit:
        """Randomly swap some pos examples with neg examples (label noise).

        Only pos/neg are swapped; neutral examples are left untouched since
        there is no well-defined "flip" for a neutral label.
        """
        import random

        if self.flip_frac <= 0.0:
            return split

        rng = random.Random(seed)
        pos, neg = list(split.pos), list(split.neg)

        n_flip = int(round(min(len(pos), len(neg)) * self.flip_frac))
        if n_flip == 0:
            return split

        pos_idx = rng.sample(range(len(pos)), n_flip)
        neg_idx = rng.sample(range(len(neg)), n_flip)

        for pi, ni in zip(pos_idx, neg_idx):
            pos[pi], neg[ni] = neg[ni], pos[pi]

        return DatasetSplit(pos=pos, neg=neg, neutral=split.neutral)

    def _build_splits(self) -> None:
        raw = self._load_raw()

        train_pos, test_pos = self._split_list(raw.pos)
        train_neg, test_neg = self._split_list(raw.neg)
        train_neutral, test_neutral = self._split_list(raw.neutral)

        train = DatasetSplit(pos=train_pos, neg=train_neg, neutral=train_neutral)
        test = DatasetSplit(pos=test_pos, neg=test_neg, neutral=test_neutral)

        self._train = self._flip_labels(train, seed=self.seed)
        self._test = self._flip_labels(test, seed=self.seed + 1) if self.flip_test else test

    @property
    def train(self) -> DatasetSplit:
        if self._train is None:
            self._build_splits()
        return self._train

    @property
    def test(self) -> DatasetSplit:
        if self._test is None:
            self._build_splits()
        return self._test

    def get_split(self, split: str) -> DatasetSplit:
        assert split in ("train", "test")
        return self.train if split == "train" else self.test

    def get(self, split: str, label: str) -> List[str]:
        assert label in ("pos", "neg", "neutral")
        return getattr(self.get_split(split), label)


class RefusalDatasetFactory(DatasetFactory):
    """Pos/neg/neutral factory built from the refusal-aird dataset.

    pos     = harmless instructions (model complies)
    neg     = harmful instructions (model refuses)
    neutral = none for this dataset (no neutral class available)

    Fixed parameters (data_path, split_filename, balance) are read from
    config/datasets.yaml under the "Refusal" key, and can be overridden
    via constructor kwargs.
    """

    CONFIG_NAME = "Refusal"

    def __init__(
        self,
        train_frac: float = 0.8,
        seed: int = 0,
        flip_frac: float = 0.0,
        flip_test: bool = False,
        **kwargs,
    ):
        super().__init__(
            train_frac=train_frac, seed=seed, flip_frac=flip_frac, flip_test=flip_test
        )
        cfg = _load_dataset_config(self.CONFIG_NAME)
        cfg.update(kwargs)

        self.data_path = cfg["data_path"]
        self.split_filename = cfg["split_filename"]
        self.balance = cfg["balance"]

    def _load_instructions(self, harmtype: str, split: str) -> List[str]:
        import json

        file_path = os.path.join(
            self.data_path, self.split_filename.format(harmtype=harmtype, split=split)
        )
        with open(file_path, "r") as f:
            data = json.load(f)
        return [d["instruction"] for d in data]

    def _load_raw(self) -> DatasetSplit:
        pos = self._load_instructions("harmless", "train")
        neg = self._load_instructions("harmful", "train")

        if self.balance:
            import random

            pos = list(pos)
            random.Random(self.seed).shuffle(pos)
            pos = pos[: len(neg)]

        return DatasetSplit(pos=pos, neg=neg, neutral=[])


class SycophancyDatasetFactory(DatasetFactory):
    """Pos/neg/neutral factory built from the Anthropic sycophancy-eval dataset.

    Each example is a (question, answer_matching_behavior,
    answer_not_matching_behavior) triple, where `question` embeds a
    user biography expressing some view and asks the model to agree/
    disagree with it.

    pos     = question + answer_matching_behavior (sycophantic response:
              agrees with the user's stated view)
    neg     = question + answer_not_matching_behavior (non-sycophantic
              response: does not simply agree with the user)
    neutral = none for this dataset

    Fixed parameters (data_path, topics, balance, max_examples_per_topic)
    are read from config/datasets.yaml under the "Sycophancy" key, and can
    be overridden via constructor kwargs.
    """

    CONFIG_NAME = "Sycophancy"

    def __init__(
        self,
        train_frac: float = 0.8,
        seed: int = 0,
        flip_frac: float = 0.0,
        flip_test: bool = False,
        **kwargs,
    ):
        super().__init__(
            train_frac=train_frac, seed=seed, flip_frac=flip_frac, flip_test=flip_test
        )
        cfg = _load_dataset_config(self.CONFIG_NAME)
        cfg.update(kwargs)

        self.data_path = cfg["data_path"]
        self.topics = cfg["topics"]
        self.max_examples_per_topic = cfg.get("max_examples_per_topic")

    def _load_topic(self, topic: str) -> List[dict]:
        import json

        file_path = os.path.join(self.data_path, f"{topic}.jsonl")
        examples = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))

        if self.max_examples_per_topic is not None:
            examples = examples[: self.max_examples_per_topic]

        return examples

    def _load_raw(self) -> DatasetSplit:
        pos, neg = [], []

        for topic in self.topics:
            for ex in self._load_topic(topic):
                question = ex["question"]
                pos.append(question + ex["answer_matching_behavior"])
                neg.append(question + ex["answer_not_matching_behavior"])

        return DatasetSplit(pos=pos, neg=neg, neutral=[])
