"""
Tensor-backed datasets for Universal Model Framework.

Uses CPUTorch tensors as the storage backend.
"""

from .dataset import Dataset


class TensorDataset(Dataset):
    """
    Dataset backed by input and target tensors.

    Example:

        dataset = TensorDataset(x, y)

        sample_x, sample_y = dataset[0]
    """

    def __init__(self, inputs, targets):
        if not hasattr(inputs, "shape") or not hasattr(inputs, "data"):
            raise TypeError(
                "inputs must be a CPUTorch Tensor."
            )

        if not hasattr(targets, "shape") or not hasattr(targets, "data"):
            raise TypeError(
                "targets must be a CPUTorch Tensor."
            )

        if len(inputs.shape) == 0:
            raise ValueError(
                "inputs must have a batch dimension."
            )

        if len(targets.shape) == 0:
            raise ValueError(
                "targets must have a batch dimension."
            )

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                "Inputs and targets must contain the same "
                "number of samples."
            )

        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(
                "TensorDataset index out of range."
            )

        input_shape = tuple(self.inputs.shape[1:])
        target_shape = tuple(self.targets.shape[1:])

        input_size = 1

        for dimension in input_shape:
            input_size *= dimension

        target_size = 1

        for dimension in target_shape:
            target_size *= dimension

        input_start = index * input_size
        input_end = input_start + input_size

        target_start = index * target_size
        target_end = target_start + target_size

        input_data = self.inputs.data[
            input_start:input_end
        ]

        target_data = self.targets.data[
            target_start:target_end
        ]

        from cputorch import Tensor

        sample_input = Tensor(
            list(input_data),
            shape=input_shape if input_shape else (1,),
            requires_grad=self.inputs.requires_grad,
        )

        sample_target = Tensor(
            list(target_data),
            shape=target_shape if target_shape else (1,),
            requires_grad=self.targets.requires_grad,
        )

        return sample_input, sample_target

    def __repr__(self):
        return (
            f"TensorDataset("
            f"samples={len(self)}, "
            f"input_shape={self.inputs.shape}, "
            f"target_shape={self.targets.shape}"
            f")"
        )


__all__ = [
    "TensorDataset",
]
