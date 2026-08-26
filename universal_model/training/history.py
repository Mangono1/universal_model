"""
Training history for Universal Model Framework.

Stores epoch-by-epoch training metrics.

The class is intentionally lightweight so it can be used by:
    - CPU-only training
    - direct tensor training
    - DataLoader training
    - custom trainers
    - future framework integrations
"""


class TrainingHistory:
    """
    Store epoch-by-epoch training results.

    Example:

        history = TrainingHistory()

        history.append(
            epoch=1,
            loss=10.5,
        )

        history.append(
            epoch=2,
            loss=5.2,
        )

        print(history.first_loss())
        print(history.latest_loss())
        print(history.best_loss())
    """

    def __init__(self):
        self.epochs = []
        self.losses = []
        self.metrics = {}

    def append(self, epoch, loss, **metrics):
        """
        Append one epoch result.

        Parameters
        ----------
        epoch:
            Epoch number.

        loss:
            Training loss.

        **metrics:
            Optional additional metrics.

        Example:
            history.append(
                epoch=1,
                loss=10.0,
                updated=2,
            )
        """

        epoch = int(epoch)
        loss = float(loss)

        self.epochs.append(epoch)
        self.losses.append(loss)

        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []

            self.metrics[name].append(float(value))

    def add(self, epoch, loss, **metrics):
        """
        Alias for append().
        """

        self.append(
            epoch=epoch,
            loss=loss,
            **metrics,
        )

    def first_loss(self):
        """
        Return the first recorded loss.

        Returns
        -------
        float or None
        """

        if not self.losses:
            return None

        return self.losses[0]

    def latest_loss(self):
        """
        Return the latest recorded loss.

        Returns
        -------
        float or None
        """

        if not self.losses:
            return None

        return self.losses[-1]

    def best_loss(self):
        """
        Return the lowest recorded loss.

        Returns
        -------
        float or None
        """

        if not self.losses:
            return None

        return min(self.losses)

    def get_metric(self, name):
        """
        Return all recorded values for a metric.

        Example:
            history.get_metric("updated")
        """

        return self.metrics.get(name, [])

    def latest_metric(self, name):
        """
        Return the latest value of a metric.

        Returns
        -------
        float or None
        """

        values = self.metrics.get(name)

        if not values:
            return None

        return values[-1]

    def first_metric(self, name):
        """
        Return the first value of a metric.

        Returns
        -------
        float or None
        """

        values = self.metrics.get(name)

        if not values:
            return None

        return values[0]

    def best_metric(self, name):
        """
        Return the lowest value of a metric.

        Returns
        -------
        float or None
        """

        values = self.metrics.get(name)

        if not values:
            return None

        return min(values)

    def epoch_loss(self, epoch):
        """
        Return loss for a specific epoch.

        Epoch numbers are the actual epoch numbers recorded
        by the Trainer, normally starting at 1.
        """

        for index, recorded_epoch in enumerate(self.epochs):
            if recorded_epoch == epoch:
                return self.losses[index]

        return None

    def to_dict(self):
        """
        Convert history into a serializable dictionary.
        """

        result = {
            "epochs": list(self.epochs),
            "losses": list(self.losses),
        }

        for name, values in self.metrics.items():
            result[name] = list(values)

        return result

    def clear(self):
        """
        Clear all recorded history.
        """

        self.epochs.clear()
        self.losses.clear()
        self.metrics.clear()

    def __len__(self):
        """
        Return number of recorded epochs.
        """

        return len(self.losses)

    def __bool__(self):
        """
        Return True when history contains records.
        """

        return bool(self.losses)

    def __repr__(self):
        return (
            f"TrainingHistory("
            f"epochs={len(self)}, "
            f"latest_loss={self.latest_loss()}"
            f")"
        )


__all__ = [
    "TrainingHistory",
]
