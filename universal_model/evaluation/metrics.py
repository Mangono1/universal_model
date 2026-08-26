"""
Evaluation metrics for Universal Model Framework.
"""

import math


def mse(prediction, target):
    """
    Mean Squared Error.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have the same shape."
        )

    total = 0.0

    for predicted, expected in zip(
        prediction.data,
        target.data,
    ):
        difference = predicted - expected
        total += difference * difference

    count = len(prediction.data)

    if count == 0:
        raise ValueError(
            "Cannot calculate MSE for empty tensors."
        )

    return total / count


def mae(prediction, target):
    """
    Mean Absolute Error.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have the same shape."
        )

    total = 0.0

    for predicted, expected in zip(
        prediction.data,
        target.data,
    ):
        total += abs(predicted - expected)

    count = len(prediction.data)

    if count == 0:
        raise ValueError(
            "Cannot calculate MAE for empty tensors."
        )

    return total / count


def rmse(prediction, target):
    """
    Root Mean Squared Error.
    """

    return math.sqrt(
        mse(prediction, target)
    )


__all__ = [
    "mse",
    "mae",
    "rmse",
]
