import numpy as np
import torch
from torch.utils.data import Dataset


class AGNewsDataset(Dataset):
    """AG News dataset with pre-tokenized DistilBERT inputs.

    Pre-tokenizes all samples at init for fast training.
    Returns (input_ids, attention_mask, label).
    """

    def __init__(self, split: str = "train", max_length: int = 128,
                 tokenizer_name: str = "distilbert-base-uncased"):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        ds = load_dataset("ag_news", split=split)

        encoded = tokenizer(
            list(ds["text"]), truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.targets = list(ds["label"])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.targets[idx]


def get_agnews(max_length: int = 128, tokenizer_name: str = "distilbert-base-uncased"):
    """Load AG News train and test splits."""
    train_dataset = AGNewsDataset("train", max_length, tokenizer_name)
    test_dataset = AGNewsDataset("test", max_length, tokenizer_name)
    return train_dataset, test_dataset


def apply_symmetric_noise_text(dataset: AGNewsDataset, noise_rate: float,
                                num_classes: int = 4, seed: int = 0):
    """Apply symmetric label noise to AG News dataset in-place."""
    if noise_rate <= 0.0:
        return dataset
    rng = np.random.RandomState(seed)
    targets = list(dataset.targets)
    for i in range(len(targets)):
        if rng.rand() < noise_rate:
            orig = targets[i]
            other = list(range(num_classes))
            other.remove(orig)
            targets[i] = int(rng.choice(other))
    dataset.targets = targets
    return dataset
