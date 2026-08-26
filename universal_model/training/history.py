"""
Training history for Universal Model Framework.
"""

class TrainingHistory:
    """
    Store metrics produced during model training.
    """

    def __init__(self):
        self.epochs = []
        self.losses = []

    def record(self, epoch, loss):
        self.epochs.append(int(epoch))
        self.losses.append(float(loss))

    def latest_loss(self):
        if not self.losses:
            return None

        return self.losses[-1]

    def first_loss(self):
        if not self.losses:
            return None

        return self.losses[0]

    def best_loss(self):
        if not self.losses:
            return None

        return min(self.losses)

    def __len__(self):
        return len(self.losses)

    def __repr__(self):
        return (
            f"TrainingHistory("
            f"epochs={len(self.epochs)}, "
            f"latest_loss={self.latest_loss()}"
            f")"
        )


__all__ = [
    "TrainingHistory",
]
