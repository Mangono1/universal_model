"""
Mean Squared Error Loss.

Universal Model Framework.
"""

from ..core.module import Module
from ..core.tensor_ops import TensorOps


class MSELoss(Module):
    """
    Mean Squared Error loss.

    Computes:

        loss = mean((prediction - target)^2)

    Both prediction and target must be CPUTorch tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        """
        Calculate mean squared error.

        Args:
            prediction: CPUTorch Tensor containing model predictions.
            target: CPUTorch Tensor containing target values.

        Returns:
            Scalar CPUTorch Tensor containing the MSE loss.
        """

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

        squared = TensorOps.multiply(
            difference,
            difference,
        )

        total = TensorOps.sum(squared)

        count = TensorOps.size(squared)

        if count <= 0:
            raise ValueError("Cannot calculate MSE for an empty tensor.")

        loss = TensorOps.multiply_scalar(
            total,
            1.0 / float(count),
        )

        return loss

    def __repr__(self):
        return "MSELoss()"


__all__ = [
    "MSELoss",
]
