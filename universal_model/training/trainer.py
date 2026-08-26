"""
Training engine for Universal Model Framework.
"""

from ..core.module import Module
from ..losses.mse import MSELoss
from ..optim.sgd import SGD

from .history import TrainingHistory


class Trainer:
    """
    Basic supervised training engine.

    Training cycle:

        forward
        loss
        zero_grad
        backward
        optimizer.step
    """

    def __init__(
        self,
        model,
        optimizer=None,
        loss_fn=None,
        verbose=True,
    ):
        if not isinstance(model, Module):
            raise TypeError(
                "Trainer model must inherit from Module."
            )

        self.model = model
        self.loss_fn = loss_fn or MSELoss()
        self.verbose = bool(verbose)

        if optimizer is None:
            optimizer = SGD(
                model,
                lr=0.001,
            )

        self.optimizer = optimizer
        self.history = TrainingHistory()

    def train_step(self, inputs, targets):
        """
        Execute one complete training step.
        """

        self.optimizer.zero_grad()

        prediction = self.model(inputs)

        loss = self.loss_fn(
            prediction,
            targets,
        )

        loss.backward()

        updated = self.optimizer.step()

        return loss, prediction, updated

    def fit(
        self,
        inputs,
        targets,
        epochs=1,
    ):
        """
        Train the model for a number of epochs.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        self.model.train()

        for epoch in range(1, epochs + 1):
            loss, prediction, updated = self.train_step(
                inputs,
                targets,
            )

            loss_value = float(
                loss.data[0]
            )

            self.history.record(
                epoch,
                loss_value,
            )

            if self.verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss_value:.6f} | "
                    f"Updated: {updated}"
                )

        return self.history

    def predict(self, inputs):
        """
        Run inference without changing parameters.
        """

        return self.model(inputs)

    def latest_loss(self):
        """
        Return latest training loss.
        """

        return self.history.latest_loss()

    def __repr__(self):
        return (
            f"Trainer("
            f"model={self.model.__class__.__name__}, "
            f"optimizer={self.optimizer}, "
            f"loss={self.loss_fn}"
            f")"
        )


__all__ = [
    "Trainer",
]
