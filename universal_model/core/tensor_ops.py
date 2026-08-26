"""
Tensor operations abstraction for Universal Model Framework.

This module isolates the model framework from the concrete CPUTorch API.

The goal is simple:

    Model layers
        ↓
    TensorOps
        ↓
    CPUTorch

If the underlying tensor engine changes in the future, most model
code should remain untouched.
"""

from typing import Sequence, Tuple

import cputorch


class TensorOps:
    """
    Small compatibility layer around CPUTorch.

    This class intentionally contains only operations that are already
    supported by the installed CPUTorch version.
    """

    Tensor = cputorch.Tensor

    @staticmethod
    def ensure_tensor(value, name: str = "value"):
        """
        Ensure that value is a CPUTorch Tensor.
        """

        if not isinstance(value, cputorch.Tensor):
            raise TypeError(
                f"{name} must be a cputorch.Tensor."
            )

        return value

    @staticmethod
    def zeros(
        shape: Sequence[int],
        requires_grad: bool = False,
    ):
        """
        Create a zero tensor.

        CPUTorch currently expects flat numerical data plus shape.
        """

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
    def ones(
        shape: Sequence[int],
        requires_grad: bool = False,
    ):
        """
        Create a tensor filled with ones.
        """

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
        """
        Create a tensor filled with one scalar value.
        """

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
        """
        Element-wise addition.
        """

        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")

        return a.add(b)

    @staticmethod
    def subtract(a, b):
        """
        Element-wise subtraction.
        """

        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")

        return a.subtract(b)

    @staticmethod
    def multiply(a, b):
        """
        Element-wise multiplication.
        """

        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")

        return a.multiply(b)

    @staticmethod
    def multiply_scalar(a, value: float):
        """
        Multiply tensor by a scalar.
        """

        TensorOps.ensure_tensor(a, "a")

        return a.multiply_scalar(float(value))

    @staticmethod
    def matmul(a, b):
        """
        Matrix multiplication.
        """

        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(b, "b")

        return a.matmul(b)

    @staticmethod
    def relu(a):
        """
        ReLU activation.
        """

        TensorOps.ensure_tensor(a, "a")

        return a.relu()

    @staticmethod
    def sum(a):
        """
        Sum all tensor elements.
        """

        TensorOps.ensure_tensor(a, "a")

        return a.sum()

    @staticmethod
    def add_bias_2d(a, bias):
        """
        Add a 2D bias tensor to a 2D activation tensor.

        CPUTorch 0.5.7 expects the bias itself to be 2D.
        """

        TensorOps.ensure_tensor(a, "a")
        TensorOps.ensure_tensor(bias, "bias")

        if a.ndim != 2:
            raise ValueError(
                "a must be a 2D tensor."
            )

        if bias.ndim != 2:
            raise ValueError(
                "bias must be a 2D tensor."
            )

        return a.add_bias_2d(bias)

    @staticmethod
    def shape(a) -> Tuple[int, ...]:
        """
        Return tensor shape as a tuple.
        """

        TensorOps.ensure_tensor(a, "a")

        return tuple(a.shape)

    @staticmethod
    def ndim(a) -> int:
        """
        Return tensor dimensionality.
        """

        TensorOps.ensure_tensor(a, "a")

        return int(a.ndim)

    @staticmethod
    def size(a) -> int:
        """
        Return number of tensor elements.
        """

        TensorOps.ensure_tensor(a, "a")

        return int(a.size)

    @staticmethod
    def backward(a):
        """
        Run backward propagation.
        """

        TensorOps.ensure_tensor(a, "a")

        return a.backward()

    @staticmethod
    def zero_grad(a):
        """
        Clear tensor gradient.
        """

        TensorOps.ensure_tensor(a, "a")

        return a.zero_grad()


__all__ = [
    "TensorOps",
]
