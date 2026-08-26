"""
Universal Model Framework - Training Engine.

Supports both direct tensor training and DataLoader-based training.

Direct:
    trainer.fit(x, target, epochs=50)

DataLoader:
    trainer.fit(loader, epochs=50)
"""

from ..data.dataloader import DataLoader


class Trainer:
    """
    Universal training engine.

    Supports:

        trainer.fit(
            inputs,
            targets,
            epochs=10,
        )

    and:

        trainer.fit(
            dataloader,
            epochs=10,
        )
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        verbose=True,
    ):
        if model is None:
            raise ValueError(
                "model must not be None."
            )

        if optimizer is None:
            raise ValueError(
                "optimizer must not be None."
            )

        if loss_fn is None:
            raise ValueError(
                "loss_fn must not be None."
            )

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.verbose = bool(verbose)

    def _train_batch(self, inputs, targets):
        """
        Execute one optimization step.
        """

        self.optimizer.zero_grad()

        prediction = self.model(inputs)

        loss = self.loss_fn(
            prediction,
            targets,
        )

        loss.backward()

        updated = self.optimizer.step()

        return loss, updated

    def _fit_direct(
        self,
        inputs,
        targets,
        epochs,
        history,
    ):
        """
        Train using direct input/target tensors.
        """

        for epoch in range(1, epochs + 1):
            loss, updated = self._train_batch(
                inputs,
                targets,
            )

            loss_value = float(
                loss.data[0]
            )

            history.append(
                epoch=epoch,
                loss=loss_value,
                updated=updated,
            )

            if self.verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {loss_value:.6f} | "
                    f"Updated: {updated}"
                )

        return history

    def _fit_loader(
        self,
        loader,
        epochs,
        history,
    ):
        """
        Train using a DataLoader.
        """

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            batch_count = 0
            total_updated = 0

            for inputs, targets in loader:
                loss, updated = self._train_batch(
                    inputs,
                    targets,
                )

                epoch_loss += float(
                    loss.data[0]
                )

                batch_count += 1
                total_updated += updated

            if batch_count == 0:
                raise ValueError(
                    "DataLoader produced zero batches."
                )

            average_loss = (
                epoch_loss / batch_count
            )

            history.append(
                epoch=epoch,
                loss=average_loss,
                updated=total_updated,
            )

            if self.verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss: {average_loss:.6f} | "
                    f"Batches: {batch_count} | "
                    f"Updated: {total_updated}"
                )

        return history

    def fit(
        self,
        inputs,
        targets=None,
        epochs=1,
        history=None,
    ):
        """
        Train the model.

        Two supported forms:

        1. Direct tensors:

            trainer.fit(
                x,
                y,
                epochs=50,
            )

        2. DataLoader:

            trainer.fit(
                loader,
                epochs=50,
            )
        """

        if not isinstance(epochs, int):
            raise TypeError(
                "epochs must be an integer."
            )

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        if history is None:
            from .history import TrainingHistory

            history = TrainingHistory()

        if isinstance(inputs, DataLoader):
            if targets is not None:
                raise ValueError(
                    "targets must be omitted when "
                    "training with DataLoader."
                )

            return self._fit_loader(
                loader=inputs,
                epochs=epochs,
                history=history,
            )

        if targets is None:
            raise ValueError(
                "targets are required when training "
                "without a DataLoader."
            )

        return self._fit_direct(
            inputs=inputs,
            targets=targets,
            epochs=epochs,
            history=history,
        )

    def train_batch(self, inputs, targets):
        """
        Train exactly one batch.
        """

        loss, updated = self._train_batch(
            inputs,
            targets,
        )

        return {
            "loss": float(loss.data[0]),
            "updated": updated,
        }

    def __repr__(self):
        return (
            f"Trainer("
            f"model={self.model.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"loss={self.loss_fn.__class__.__name__}, "
            f"verbose={self.verbose}"
            f")"
        )


__all__ = [
    "Trainer",
]
