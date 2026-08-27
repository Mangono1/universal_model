"""
Universal Model Framework - Training Engine.

Supports:
    - Direct tensor training
    - DataLoader training
    - Continuing training from existing TrainingHistory
    - Checkpoint resume workflows

CPU-first and dependency-free.
"""

from ..data.dataloader import DataLoader


class Trainer:
    """
    Universal training engine.

    Direct tensors:

        trainer.fit(
            inputs,
            targets,
            epochs=10,
        )

    DataLoader:

        trainer.fit(
            loader,
            epochs=10,
        )

    Resume:

        history = checkpoint.training_history()

        trainer.fit(
            x,
            y,
            epochs=10,
            history=history,
        )

    When an existing history is supplied, training continues from
    the last recorded epoch instead of resetting the epoch counter.
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

    # ---------------------------------------------------------
    # Internal training
    # ---------------------------------------------------------

    def _train_batch(self, inputs, targets):
        """
        Execute exactly one optimization step.
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

    def _next_epoch(self, history):
        """
        Return the next epoch number.

        Existing history is treated as the authoritative record
        of previous training progress.
        """

        if history is None:
            return 1

        if not history:
            return 1

        return int(history.epochs[-1]) + 1

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

        start_epoch = self._next_epoch(history)

        for offset in range(epochs):
            epoch = start_epoch + offset

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

        start_epoch = self._next_epoch(history)

        for offset in range(epochs):
            epoch = start_epoch + offset

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

    # ---------------------------------------------------------
    # Public training API
    # ---------------------------------------------------------

    def fit(
        self,
        inputs,
        targets=None,
        epochs=1,
        history=None,
    ):
        """
        Train the model.

        Parameters
        ----------
        inputs:
            Input Tensor or DataLoader.

        targets:
            Target Tensor for direct tensor training.
            Must be omitted when using DataLoader.

        epochs:
            Number of NEW epochs to train.

        history:
            Optional existing TrainingHistory.

            If supplied, training continues after the last
            recorded epoch.

        Examples
        --------

        Fresh training:

            trainer.fit(
                x,
                y,
                epochs=10,
            )

        Resume:

            restored_history = checkpoint.training_history()

            trainer.fit(
                x,
                y,
                epochs=10,
                history=restored_history,
            )

        The second example records epochs 11-20 when the
        restored history already contains epochs 1-10.
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

        else:
            if not hasattr(history, "append"):
                raise TypeError(
                    "history must be a TrainingHistory object."
                )

            if not hasattr(history, "epochs"):
                raise TypeError(
                    "history must contain epoch records."
                )

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

    def train_batch(
        self,
        inputs,
        targets,
    ):
        """
        Train exactly one batch.
        """

        loss, updated = self._train_batch(
            inputs,
            targets,
        )

        return {
            "loss": float(
                loss.data[0]
            ),
            "updated": updated,
        }

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

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
