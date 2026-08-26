"""
Evaluation engine for Universal Model Framework.
"""

from .metrics import mse, mae, rmse


class EvaluationResult:
    """
    Structured result returned by Evaluator.
    """

    def __init__(
        self,
        loss,
        mse_value,
        mae_value,
        rmse_value,
        samples,
        batches,
    ):
        self.loss = float(loss)
        self.mse = float(mse_value)
        self.mae = float(mae_value)
        self.rmse = float(rmse_value)
        self.samples = int(samples)
        self.batches = int(batches)

    def to_dict(self):
        return {
            "loss": self.loss,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "samples": self.samples,
            "batches": self.batches,
        }

    def __repr__(self):
        return (
            "EvaluationResult("
            f"loss={self.loss}, "
            f"mse={self.mse}, "
            f"mae={self.mae}, "
            f"rmse={self.rmse}, "
            f"samples={self.samples}, "
            f"batches={self.batches}"
            ")"
        )


class Evaluator:
    """
    Evaluate a trained model using direct tensors or DataLoader.
    """

    def __init__(self, model, loss_fn):
        if model is None:
            raise ValueError(
                "model must not be None."
            )

        if loss_fn is None:
            raise ValueError(
                "loss_fn must not be None."
            )

        self.model = model
        self.loss_fn = loss_fn

    def _evaluate_batch(self, inputs, targets):
        prediction = self.model(inputs)

        loss = self.loss_fn(
            prediction,
            targets,
        )

        return (
            prediction,
            targets,
            loss,
        )

    def evaluate(
        self,
        inputs,
        targets=None,
    ):
        """
        Evaluate directly or through DataLoader.
        """

        from ..data.dataloader import DataLoader

        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        total_samples = 0
        total_batches = 0

        if isinstance(inputs, DataLoader):
            if targets is not None:
                raise ValueError(
                    "targets must be omitted when using DataLoader."
                )

            for batch_inputs, batch_targets in inputs:
                (
                    prediction,
                    batch_targets,
                    loss,
                ) = self._evaluate_batch(
                    batch_inputs,
                    batch_targets,
                )

                batch_count = batch_targets.shape[0]

                total_loss += float(
                    loss.data[0]
                )

                total_mse += mse(
                    prediction,
                    batch_targets,
                )

                total_mae += mae(
                    prediction,
                    batch_targets,
                )

                total_rmse += rmse(
                    prediction,
                    batch_targets,
                )

                total_samples += batch_count
                total_batches += 1

        else:
            if targets is None:
                raise ValueError(
                    "targets are required for direct evaluation."
                )

            (
                prediction,
                targets,
                loss,
            ) = self._evaluate_batch(
                inputs,
                targets,
            )

            total_loss = float(
                loss.data[0]
            )

            total_mse = mse(
                prediction,
                targets,
            )

            total_mae = mae(
                prediction,
                targets,
            )

            total_rmse = rmse(
                prediction,
                targets,
            )

            total_samples = targets.shape[0]
            total_batches = 1

        if total_batches == 0:
            raise ValueError(
                "Evaluation produced zero batches."
            )

        if isinstance(inputs, DataLoader):
            average_loss = (
                total_loss / total_batches
            )

            average_mse = (
                total_mse / total_batches
            )

            average_mae = (
                total_mae / total_batches
            )

            average_rmse = (
                total_rmse / total_batches
            )
        else:
            average_loss = total_loss
            average_mse = total_mse
            average_mae = total_mae
            average_rmse = total_rmse

        return EvaluationResult(
            loss=average_loss,
            mse_value=average_mse,
            mae_value=average_mae,
            rmse_value=average_rmse,
            samples=total_samples,
            batches=total_batches,
        )


__all__ = [
    "Evaluator",
    "EvaluationResult",
]
