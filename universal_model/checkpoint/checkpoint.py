"""
Training checkpoint support.

Universal Model Framework.

Stores model state together with training progress.
CPU-first and dependency-free.
"""

import json
from pathlib import Path

from ..state import state_dict, load_state_dict, StateDict


CHECKPOINT_FORMAT = "universal_model_checkpoint"
CHECKPOINT_VERSION = 1


class TrainingCheckpoint:
    """
    Container for resumable training state.

    A checkpoint contains:

        - model state
        - optimizer state
        - epoch
        - training history
        - metadata

    The object is intentionally independent from any specific model
    architecture so it can be used by different Universal Model models.
    """

    def __init__(
        self,
        model_state=None,
        optimizer_state=None,
        epoch=0,
        history=None,
        metadata=None,
    ):
        self.model_state = (
            model_state
            if isinstance(model_state, StateDict)
            else StateDict(model_state or {})
        )

        self.optimizer_state = dict(
            optimizer_state or {}
        )

        self.epoch = int(epoch)

        self.history = dict(
            history or {}
        )

        self.metadata = dict(
            metadata or {}
        )

    # ---------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------

    @property
    def model_parameters(self):
        """
        Return the number of model Parameter objects.
        """

        return len(self.model_state)

    @property
    def optimizer_parameters(self):
        """
        Return the number of optimizer-managed parameters.

        This uses the stored optimizer metadata when available.
        """

        value = self.optimizer_state.get(
            "parameter_count",
            0,
        )

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @property
    def model_scalars(self):
        """
        Return the total number of scalar values stored in the model.

        StateDict values are expected to contain tensors.
        """

        total = 0

        for value in self.model_state.values.values():
            if hasattr(value, "numel"):
                try:
                    total += int(value.numel())
                    continue
                except Exception:
                    pass

            if hasattr(value, "size"):
                try:
                    total += int(value.size)
                    continue
                except Exception:
                    pass

            if hasattr(value, "data"):
                try:
                    total += len(value.data)
                    continue
                except Exception:
                    pass

            try:
                total += len(value)
            except Exception:
                pass

        return total

    @property
    def history_entries(self):
        """
        Return the number of recorded history entries.

        The normal TrainingHistory representation stores an
        'epochs' list, so that is used as the authoritative count.
        """

        epochs = self.history.get("epochs")

        if isinstance(epochs, (list, tuple)):
            return len(epochs)

        losses = self.history.get("losses")

        if isinstance(losses, (list, tuple)):
            return len(losses)

        return 0

    @property
    def optimizer_type(self):
        """
        Return the optimizer class name stored in the checkpoint.
        """

        return self.optimizer_state.get(
            "type"
        )

    @property
    def learning_rate(self):
        """
        Return the stored optimizer learning rate.
        """

        value = self.optimizer_state.get(
            "lr"
        )

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @property
    def latest_loss(self):
        """
        Return the latest recorded training loss.
        """

        losses = self.history.get("losses")

        if not losses:
            return None

        return float(losses[-1])

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(self):
        """
        Convert checkpoint to a serializable dictionary.
        """

        return {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "epoch": self.epoch,
            "model": dict(
                self.model_state.values
            ),
            "optimizer": dict(
                self.optimizer_state
            ),
            "history": dict(
                self.history
            ),
            "metadata": dict(
                self.metadata
            ),
        }

    @classmethod
    def from_dict(cls, data):
        """
        Construct checkpoint from dictionary.
        """

        if not isinstance(data, dict):
            raise TypeError(
                "Checkpoint data must be a dictionary."
            )

        if data.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                "Invalid training checkpoint format."
            )

        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported training checkpoint version."
            )

        return cls(
            model_state=StateDict(
                data.get("model", {})
            ),
            optimizer_state=data.get(
                "optimizer",
                {},
            ),
            epoch=data.get(
                "epoch",
                0,
            ),
            history=data.get(
                "history",
                {},
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    # ---------------------------------------------------------
    # Disk operations
    # ---------------------------------------------------------

    def save(self, path):
        """
        Save checkpoint to disk.
        """

        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
            )

        return destination

    @classmethod
    def load(cls, path):
        """
        Load checkpoint from disk.
        """

        source = Path(path)

        if not source.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {source}"
            )

        with source.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        return cls.from_dict(data)

    # ---------------------------------------------------------
    # Restore
    # ---------------------------------------------------------

    def restore_model(
        self,
        model,
        strict=True,
    ):
        """
        Restore model parameters from this checkpoint.
        """

        return load_state_dict(
            model,
            self.model_state,
            strict=strict,
        )

    def restore_optimizer(
        self,
        optimizer,
    ):
        """
        Restore basic optimizer configuration.

        Current checkpoint format stores optimizer metadata
        rather than optimizer-internal state.

        This restores the learning rate when available.
        """

        if optimizer is None:
            raise ValueError(
                "optimizer must not be None."
            )

        restored = 0

        if self.learning_rate is not None:
            if hasattr(optimizer, "lr"):
                optimizer.lr = self.learning_rate
                restored = 1

        return restored

    def restore(
        self,
        model,
        optimizer=None,
        strict=True,
    ):
        """
        Restore model and optional optimizer state.

        Returns a dictionary describing what was restored.
        """

        model_result = self.restore_model(
            model,
            strict=strict,
        )

        optimizer_result = 0

        if optimizer is not None:
            optimizer_result = self.restore_optimizer(
                optimizer
            )

        return {
            "model": model_result,
            "optimizer": optimizer_result,
            "epoch": self.epoch,
            "history_entries": self.history_entries,
        }

    # ---------------------------------------------------------
    # History
    # ---------------------------------------------------------

    def training_history(self):
        """
        Reconstruct a TrainingHistory object.

        This avoids forcing callers to manually convert the
        serialized history dictionary.
        """

        from ..training.history import TrainingHistory

        result = TrainingHistory()

        epochs = self.history.get(
            "epochs",
            [],
        )

        losses = self.history.get(
            "losses",
            [],
        )

        metric_names = [
            name
            for name in self.history
            if name not in (
                "epochs",
                "losses",
            )
        ]

        count = min(
            len(epochs),
            len(losses),
        )

        for index in range(count):
            metrics = {}

            for name in metric_names:
                values = self.history.get(
                    name,
                    [],
                )

                if index < len(values):
                    metrics[name] = values[index]

            result.append(
                epoch=epochs[index],
                loss=losses[index],
                **metrics,
            )

        return result

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self):
        return (
            f"TrainingCheckpoint("
            f"epoch={self.epoch}, "
            f"model_parameters={self.model_parameters}, "
            f"model_scalars={self.model_scalars}, "
            f"optimizer_parameters={self.optimizer_parameters}, "
            f"history_entries={self.history_entries}"
            f")"
        )


def create_checkpoint(
    model,
    optimizer=None,
    epoch=0,
    history=None,
    metadata=None,
):
    """
    Create a training checkpoint.
    """

    optimizer_state = {}

    if optimizer is not None:
        optimizer_state = {
            "type": optimizer.__class__.__name__,
            "lr": float(
                getattr(
                    optimizer,
                    "lr",
                    0.0,
                )
            ),
            "parameter_count": int(
                optimizer.parameter_count()
            ),
        }

    if history is None:
        history_data = {}

    elif hasattr(history, "to_dict"):
        history_data = history.to_dict()

    elif isinstance(history, dict):
        history_data = dict(history)

    else:
        raise TypeError(
            "history must be a TrainingHistory "
            "or dictionary."
        )

    return TrainingCheckpoint(
        model_state=state_dict(model),
        optimizer_state=optimizer_state,
        epoch=epoch,
        history=history_data,
        metadata=metadata,
    )


def save_checkpoint(
    model,
    path,
    optimizer=None,
    epoch=0,
    history=None,
    metadata=None,
):
    """
    Create and save a training checkpoint.
    """

    checkpoint = create_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        history=history,
        metadata=metadata,
    )

    return checkpoint.save(path)


def load_checkpoint(
    path,
):
    """
    Load a training checkpoint.
    """

    return TrainingCheckpoint.load(path)


__all__ = [
    "TrainingCheckpoint",
    "create_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
]
