"""
Universal Model Framework - Dataset Base Classes.

Datasets define how training data is accessed.
They do not know anything about models, optimizers, or CUDA.
"""

from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Base dataset interface.

    A dataset must provide:
        - __len__()
        - __getitem__(index)

    The returned item is normally:
        (input, target)
    """

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class Subset(Dataset):
    """
    View of another dataset using selected indices.
    """

    def __init__(self, dataset, indices):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "Subset requires a Dataset instance."
            )

        self.dataset = dataset
        self.indices = list(indices)

        for index in self.indices:
            if not isinstance(index, int):
                raise TypeError(
                    "Subset indices must be integers."
                )

            if index < 0 or index >= len(dataset):
                raise IndexError(
                    f"Dataset index out of range: {index}"
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __repr__(self):
        return (
            f"Subset("
            f"size={len(self)}, "
            f"dataset_size={len(self.dataset)}"
            f")"
        )


__all__ = [
    "Dataset",
    "Subset",
]
