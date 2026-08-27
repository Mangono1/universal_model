"""
Mean Absolute Error Loss.

Universal Model Framework.
"""

from ..core.module import Module
from ..core.tensor_ops import TensorOps


class MAELoss(Module):
    """
    Mean Absolute Error loss.

    Computes:

        loss = mean(abs(prediction - target))

    Both prediction and target must be CPUTorch tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        if prediction is None:
            raise ValueError("prediction must not be None.")

        if target is None:
            raise ValueError("target must not be None.")

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

        # CPUTorch currently has no abs() operation.
        # Build |x| using x * sign(x).
        values = difference.data
        absolute_values = []

        for value in values:
            absolute_values.append(
                abs(float(value))
            )

        absolute = TensorOps.from_data(
            absolute_values,
            shape=difference.shape,
            requires_grad=False,
        )

        total = TensorOps.sum(absolute)

        count = TensorOps.size(absolute)

        if count <= 0:
            raise ValueError(
                "Cannot calculate MAE for an empty tensor."
            )

        return TensorOps.multiply_scalar(
            total,
            1.0 / float(count),
        )

    def __repr__(self):
        return "MAELoss()"


__all__ = [
    "MAELoss",
]
