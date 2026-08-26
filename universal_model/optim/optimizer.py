"""
Base optimizer classes.

Universal Model Framework.
"""

from ..core.module import Module


class Optimizer:
    """
    Base class for parameter optimizers.

    Optimizers receive a model or iterable of parameters and update
    trainable parameters using their gradients.
    """

    def __init__(self, parameters, lr=0.001):
        if lr <= 0:
            raise ValueError("Learning rate must be greater than zero.")

        self.lr = float(lr)
        self._parameters = []

        if isinstance(parameters, Module):
            parameters = parameters.parameters()

        for parameter in parameters:
            if parameter is None:
                continue

            if not hasattr(parameter, "tensor"):
                raise TypeError(
                    "Optimizer parameters must be Parameter objects."
                )

            self._parameters.append(parameter)

    @property
    def parameters(self):
        """
        Return optimizer parameters.
        """
        return self._parameters

    def zero_grad(self):
        """
        Clear gradients of all managed parameters.
        """

        for parameter in self._parameters:
            parameter.zero_grad()

    def step(self):
        """
        Update parameters.

        Subclasses must implement this method.
        """

        raise NotImplementedError(
            "Optimizer subclasses must implement step()."
        )

    def parameter_count(self):
        """
        Return total number of scalar parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self._parameters
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"lr={self.lr}, "
            f"parameters={len(self._parameters)}"
            f")"
        )


__all__ = [
    "Optimizer",
]
