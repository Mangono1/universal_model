"""
Tensor operations abstraction for Universal Model Framework.

Universal Model Framework
"""

from typing import Sequence, Tuple

import cputorch


class TensorOps:
    """
    Compatibility layer around CPUTorch.
    """

    Tensor = cputorch.Tensor

    @staticmethod
    def ensure_tensor(value, name: str = "value"):
        if not isinstance(value, cputorch.Tensor):
            raise TypeError(
                f"{name} must be a cputorch.Tensor."
            )
        return value

    @staticmethod
    def zeros(shape: Sequence[int], requires_grad: bool = False):
        shape = tuple(int(x) for x in shape)

        if not shape:
            raise ValueError("shape must not be empty.")

        size = 1
        for dimension in shape:
            if dimension <= 0:
                raise ValueError(
                    "all dimensions must be greater than zero."
                )
            size *= dimension

        return cputorch.Tensor(
            [0.0] * size,
            shape=shape,
            requires_grad=requires_grad,
        )

    @staticmethod
    def ones(shape: Sequence[int], requires_grad: bool = False):
        shape = tuple(int(x) for x in shape)

        if not shape:
            raise ValueError("shape must not be empty.")

        size = 1
        for dimension in shape:
            if dimension <= 0:
                raise ValueError(
                    "all dimensions must be greater than zero."
                )
            size *= dimension

        return cputorch.Tensor(
            [1.0] * size,
            shape=shape,
            requires_grad=requires_grad,
        )

    @staticmethod
    def full(
        shape: Sequence[int],
        value: float,
        requires_grad: bool = False,
    ):
        shape = tuple(int(x) for x in shape)

        if not shape:
            raise ValueError("shape must not be empty.")

        size = 1
        for dimension in shape:
            if dimension <= 0:
                raise ValueError(
                    "all dimensions must be greater than zero."
                )
            size *= dimension

        return cputorch.Tensor(
            [float(value)] * size,
            shape=shape,
            requires_grad=requires_grad,
        )

    @staticmethod
    def add(a, b):
        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")
        return a.add(b)

    @staticmethod
    def subtract(a, b):
        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")
        return a.subtract(b)

    @staticmethod
    def multiply(a, b):
        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")
        return a.multiply(b)

    @staticmethod
    def multiply_scalar(a, value: float):
        TensorOps.ensure_tensor(a, "a")
        return a.multiply_scalar(float(value))

    @staticmethod
    def matmul(a, b):
        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")
        return a.matmul(b)

    @staticmethod
    def relu(a):
        TensorOps.ensure_tensor(a, "a")
        return a.relu()

    @staticmethod
    def sum(a):
        TensorOps.ensure_tensor(a, "a")
        return a.sum()

    @staticmethod
    def add_bias_2d(a, bias):
        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(bias, "bias")

        if a.ndim != 2:
            raise ValueError("a must be a 2D tensor.")

        if bias.ndim != 2:
            raise ValueError("bias must be a 2D tensor.")

        return a.add_bias_2d(bias)

    @staticmethod
    def shape(a) -> Tuple[int, ...]:
        TensorOps.ensure_tensor(a, "a")
        return tuple(a.shape)

    @staticmethod
    def ndim(a) -> int:
        TensorOps.ensure_tensor(a, "a")
        return int(a.ndim)

    @staticmethod
    def size(a) -> int:
        TensorOps.ensure_tensor(a, "a")
        return int(a.size)

    @staticmethod
    def item(a, index: int = 0) -> float:
        """
        Extract one scalar value from a CPUTorch tensor.
        """

        TensorOps.ensure_tensor(a, "a")

        size = TensorOps.size(a)

        if size <= 0:
            raise ValueError(
                "Cannot extract an item from an empty tensor."
            )

        index = int(index)

        if index < 0 or index >= size:
            raise IndexError(
                f"Tensor item index {index} out of range "
                f"for tensor with {size} elements."
            )

        return float(a.item(index))

    @staticmethod
    def backward(a):
        TensorOps.ensure_tensor(a, "a")
        return a.backward()

    @staticmethod
    def zero_grad(a):
        TensorOps.ensure_tensor(a, "a")
        return a.zero_grad()


__all__ = [
    "TensorOps",
]
