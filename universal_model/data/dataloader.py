"""
Universal Model Framework - DataLoader.

Provides batching and shuffling on top of Dataset objects.
"""

import random

from .dataset import Dataset


class DataLoader:
    """
    Iterate over a Dataset in batches.

    Example:

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
        )
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        seed=None,
    ):
        if not isinstance(dataset, Dataset):
            raise TypeError(
                "DataLoader requires a Dataset instance."
            )

        if not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed

    def __len__(self):
        size = len(self.dataset)

        if self.drop_last:
            return size // self.batch_size

        return (
            size + self.batch_size - 1
        ) // self.batch_size

    def _indices(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            generator = random.Random(self.seed)
            generator.shuffle(indices)

        return indices

    def __iter__(self):
        indices = self._indices()

        batch = []

        for index in indices:
            batch.append(self.dataset[index])

            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []

        if batch and not self.drop_last:
            yield self._collate(batch)

    @staticmethod
    def _collate(samples):
        if not samples:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        first = samples[0]

        if not isinstance(first, tuple):
            return samples

        if len(first) != 2:
            raise ValueError(
                "Dataset samples must contain "
                "(input, target)."
            )

        inputs = [sample[0] for sample in samples]
        targets = [sample[1] for sample in samples]

        return (
            DataLoader._stack_tensors(inputs),
            DataLoader._stack_tensors(targets),
        )

    @staticmethod
    def _stack_tensors(tensors):
        if not tensors:
            raise ValueError(
                "Cannot stack an empty tensor list."
            )

        first = tensors[0]

        if not hasattr(first, "shape") or not hasattr(first, "data"):
            raise TypeError(
                "DataLoader tensor batching requires "
                "CPUTorch Tensor objects."
            )

        sample_shape = tuple(first.shape)

        for tensor in tensors:
            if tuple(tensor.shape) != sample_shape:
                raise ValueError(
                    "All tensors in a batch must have "
                    "the same shape."
                )

        data = []

        for tensor in tensors:
            data.extend(tensor.data)

        from cputorch import Tensor

        return Tensor(
            data,
            shape=(len(tensors),) + sample_shape,
            requires_grad=any(
                tensor.requires_grad
                for tensor in tensors
            ),
        )

    def __repr__(self):
        return (
            f"DataLoader("
            f"dataset={len(self.dataset)}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"drop_last={self.drop_last}, "
            f"batches={len(self)}"
            f")"
        )


__all__ = [
    "DataLoader",
]
