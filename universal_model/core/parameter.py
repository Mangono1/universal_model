"""
Trainable parameter abstraction for Universal Model Framework.
"""

from typing import Any


class Parameter:
    """
    Wrapper around a tensor that represents a trainable model parameter.
    """

    def __init__(self, tensor: Any, requires_grad: bool = True):
        self.tensor = tensor
        self.requires_grad = bool(requires_grad)

        # Keep the underlying tensor consistent with Parameter.
        if hasattr(self.tensor, "requires_grad"):
            try:
                self.tensor.requires_grad = self.requires_grad
            except Exception:
                pass

    @property
    def grad(self):
        """Return the gradient of the underlying tensor."""
        return getattr(self.tensor, "grad", None)

    @property
    def shape(self):
        """Return the tensor shape."""
        return getattr(self.tensor, "shape", None)

    @property
    def ndim(self):
        """Return the tensor dimensionality."""
        return getattr(self.tensor, "ndim", None)

    @property
    def size(self):
        """Return the number of elements."""
        value = getattr(self.tensor, "size", None)

        if callable(value):
            return value()

        return value

    def numel(self) -> int:
        """
        Return the number of scalar elements in this parameter.
        """

        size = self.size

        if size is not None:
            return int(size)

        shape = self.shape

        if shape is None:
            return 0

        result = 1

        for dimension in shape:
            result *= int(dimension)

        return result

    def zero_grad(self) -> None:
        """
        Clear the parameter gradient when supported by CPUTorch.
        """

        if hasattr(self.tensor, "zero_grad"):
            self.tensor.zero_grad()

    def __repr__(self) -> str:
        return (
            f"Parameter("
            f"tensor={self.tensor}, "
            f"requires_grad={self.requires_grad}"
            f")"
        )
