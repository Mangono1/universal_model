"""
Base optimizer classes for Universal Model Framework.
"""

from ..core.module import Module


class OptimizerStepResult:
    """
    Result returned by an optimizer step.

    Keeps track of both:
    - parameter tensors updated
    - scalar values updated
    """

    def __init__(
        self,
        updated_parameters=0,
        updated_scalars=0,
    ):
        self.updated_parameters = int(
            updated_parameters
        )

        self.updated_scalars = int(
            updated_scalars
        )

    @property
    def parameters(self):
        """
        Number of parameter tensors updated.
        """

        return self.updated_parameters

    @property
    def scalars(self):
        """
        Number of scalar parameter values updated.
        """

        return self.updated_scalars

    def __int__(self):
        """
        Backward compatibility with code expecting
        the old integer result.
        """

        return self.updated_parameters

    def __repr__(self):
        return (
            "OptimizerStepResult("
            f"updated_parameters={self.updated_parameters}, "
            f"updated_scalars={self.updated_scalars}"
            ")"
        )


class Optimizer:
    """
    Base class for parameter optimizers.

    Optimizers receive a model or iterable of parameters
    and update trainable parameters using their gradients.
    """

    def __init__(
        self,
        parameters,
        lr=0.001,
    ):
        if lr <= 0:
            raise ValueError(
                "Learning rate must be greater than zero."
            )

        self.lr = float(lr)
        self._parameters = []

        if isinstance(parameters, Module):
            parameters = parameters.parameters()

        for parameter in parameters:
            if parameter is None:
                continue

            if not hasattr(parameter, "tensor"):
                raise TypeError(
                    "Optimizer parameters must be "
                    "Parameter objects."
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

    def parameter_tensor_count(self):
        """
        Return number of Parameter objects managed.
        """

        return len(self._parameters)

    def __len__(self):
        """
        Return number of Parameter objects managed.
        """

        return len(self._parameters)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"lr={self.lr}, "
            f"parameters={len(self._parameters)}, "
            f"scalars={self.parameter_count()}"
            f")"
        )


__all__ = [
    "Optimizer",
    "OptimizerStepResult",
]