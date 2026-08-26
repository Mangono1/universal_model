"""
Activation layers for Universal Model Framework.
"""

import cputorch

from ..core.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit.

    Computes:

        y = max(0, x)

    The actual numerical operation is delegated to CPUTorch.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not isinstance(x, cputorch.Tensor):
            raise TypeError(
                "ReLU input must be a cputorch.Tensor."
            )

        return x.relu()

    def __repr__(self):
        return "ReLU()"
