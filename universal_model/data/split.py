"""
Universal Model Framework - Dataset splitting utilities.
"""

import random

from .dataset import Dataset, Subset


def train_test_split(
    dataset,
    train_ratio=0.8,
    seed=None,
    shuffle=True,
):
    """
    Split a dataset into train and test subsets.

    Returns:
        train_dataset, test_dataset
    """

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "dataset must be a Dataset instance."
        )

    if not isinstance(train_ratio, (int, float)):
        raise TypeError(
            "train_ratio must be numeric."
        )

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError(
            "train_ratio must be greater than 0 "
            "and less than 1."
        )

    indices = list(range(len(dataset)))

    if shuffle:
        generator = random.Random(seed)
        generator.shuffle(indices)

    split_index = int(
        len(indices) * train_ratio
    )

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, test_indices),
    )


def train_validation_split(
    dataset,
    train_ratio=0.8,
    seed=None,
    shuffle=True,
):
    """
    Split a dataset into training and validation subsets.
    """

    return train_test_split(
        dataset=dataset,
        train_ratio=train_ratio,
        seed=seed,
        shuffle=shuffle,
    )


__all__ = [
    "train_test_split",
    "train_validation_split",
]
