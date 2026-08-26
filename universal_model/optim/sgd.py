"""
Stochastic Gradient Descent optimizer.

Universal Model Framework.
"""

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Update rule:

        parameter = parameter - lr * gradient
    """

    def __init__(self, parameters, lr=0.001):
        super().__init__(
            parameters=parameters,
            lr=lr,
        )

    def step(self):
        """
        Apply one SGD parameter update.
        """

        updated = 0

        for parameter in self.parameters:
            tensor = parameter.tensor
            gradient = tensor.grad

            if gradient is None:
                continue

            for index in range(tensor.size):
                new_value = (
                    tensor.data[index]
                    - self.lr * gradient.data[index]
                )

                tensor.set_item(
                    index,
                    new_value,
                )

            updated += 1

        return updated

    def __repr__(self):
        return (
            f"SGD("
            f"lr={self.lr}, "
            f"parameters={len(self.parameters)}"
            f")"
        )


__all__ = [
    "SGD",
]
