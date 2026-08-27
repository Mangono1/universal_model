"""
Mean Squared Error Loss.

Universal Model Framework.
"""

from ..core.module import Module
from ..core.tensor_ops import TensorOps


class MSELoss(Module):
    """
    Mean Squared Error loss.

    loss = mean((prediction - target)^2)
    """

    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction not in ("mean", "sum"):
            raise ValueError(
                "reduction must be 'mean' or 'sum'."
            )

        self.reduction = reduction

    def forward(self, prediction, target):
        if prediction is None:
            raise ValueError("prediction must not be None.")

        if target is None:
            raise ValueError("target must not be None.")

        prediction = TensorOps.ensure_tensor(
            prediction,
            "prediction",
        )

        target = TensorOps.ensure_tensor(
            target,
            "target",
        )

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must have the same shape. "
                f"Got prediction={prediction.shape}, "
                f"target={target.shape}."
            )

        difference = TensorOps.subtract(
            prediction,
            target,
        )

        squared = TensorOps.multiply(
            difference,
            difference,
        )

        total = TensorOps.sum(squared)

        if self.reduction == "sum":
            return total

        count = TensorOps.size(squared)

        if count <= 0:
            raise ValueError(
                "Cannot calculate MSE for an empty tensor."
            )

        return TensorOps.multiply_scalar(
            total,
            1.0 / float(count),
        )

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"


__all__ = [
    "MSELoss",
]
