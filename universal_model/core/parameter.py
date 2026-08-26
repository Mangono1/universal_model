"""
Parameter abstraction.

The actual tensor engine remains CPUTorch.
"""

from typing import Any


class Parameter:
    """
    Wrapper around a CPUTorch tensor intended for optimization.
    """

    def __init__(self, tensor: Any, requires_grad: bool = True):
        self.tensor = tensor
        self.requires_grad = requires_grad

    def zero_grad(self) -> None:
        if hasattr(self.tensor, "zero_grad"):
            self.tensor.zero_grad()

    def __repr__(self) -> str:
        return (
            f"Parameter("
            f"tensor={self.tensor!r}, "
            f"requires_grad={self.requires_grad}"
            f")"
        )
