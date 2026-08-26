"""
Core module abstraction for Universal Model Framework.
"""

from typing import Dict, Iterator, Any


class Module:
    """
    Base class for all Universal Model components.

    A Module can contain:
        - trainable parameters
        - child modules
        - nested module hierarchies

    This abstraction intentionally stays independent from any
    particular neural-network architecture.
    """

    def __init__(self):
        self._parameters: Dict[str, object] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training = True

    # ---------------------------------------------------------
    # Parameter registration
    # ---------------------------------------------------------

    def register_parameter(self, name: str, parameter: object) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter name must be a non-empty string.")

        if parameter is None:
            raise ValueError("Parameter cannot be None.")

        self._parameters[name] = parameter

    # ---------------------------------------------------------
    # Child module registration
    # ---------------------------------------------------------

    def add_module(self, name: str, module: "Module") -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Module name must be a non-empty string.")

        if not isinstance(module, Module):
            raise TypeError("Child module must inherit from Module.")

        self._modules[name] = module

    # ---------------------------------------------------------
    # Parameter iteration
    # ---------------------------------------------------------

    def parameters(self) -> Iterator[object]:
        """
        Yield all parameters recursively.
        """

        for parameter in self._parameters.values():
            yield parameter

        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = ""):
        """
        Yield (name, parameter) recursively.
        """

        for name, parameter in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, parameter

        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(module_prefix)

    # ---------------------------------------------------------
    # Parameter counting
    # ---------------------------------------------------------

    def parameter_count(self) -> int:
        """
        Return the total number of scalar trainable parameters.
        """

        total = 0

        for parameter in self.parameters():
            if hasattr(parameter, "numel"):
                total += int(parameter.numel())
            else:
                tensor = getattr(parameter, "tensor", None)

                if tensor is None:
                    continue

                size = getattr(tensor, "size", 0)

                if callable(size):
                    size = size()

                if size is not None:
                    total += int(size)

        return total

    def trainable_parameter_count(self) -> int:
        """
        Count only parameters with requires_grad=True.
        """

        total = 0

        for parameter in self.parameters():
            if not getattr(parameter, "requires_grad", False):
                continue

            if hasattr(parameter, "numel"):
                total += int(parameter.numel())

        return total

    # ---------------------------------------------------------
    # State dictionary
    # ---------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """
        Return a flat dictionary containing model parameter tensors.

        Keys use the same hierarchical naming convention as
        named_parameters().
        """

        state = {}

        for name, parameter in self.named_parameters():
            state[name] = getattr(parameter, "tensor", parameter)

        return state

    # ---------------------------------------------------------
    # Load state dictionary
    # ---------------------------------------------------------

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> None:
        """
        Load parameter tensors from a state dictionary.

        The current implementation validates names and shapes,
        then replaces the parameter tensor references.
        """

        current = dict(self.named_parameters())

        incoming_names = set(state_dict.keys())
        current_names = set(current.keys())

        missing = sorted(current_names - incoming_names)
        unexpected = sorted(incoming_names - current_names)

        if strict and (missing or unexpected):
            message_parts = []

            if missing:
                message_parts.append(
                    f"missing keys: {missing}"
                )

            if unexpected:
                message_parts.append(
                    f"unexpected keys: {unexpected}"
                )

            raise ValueError(
                "State dictionary mismatch: "
                + "; ".join(message_parts)
            )

        for name, tensor in state_dict.items():
            if name not in current:
                continue

            parameter = current[name]

            existing_tensor = getattr(parameter, "tensor", None)

            existing_shape = getattr(existing_tensor, "shape", None)
            incoming_shape = getattr(tensor, "shape", None)

            if (
                existing_shape is not None
                and incoming_shape is not None
                and tuple(existing_shape) != tuple(incoming_shape)
            ):
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"expected {existing_shape}, "
                    f"got {incoming_shape}."
                )

            parameter.tensor = tensor

    # ---------------------------------------------------------
    # Training / evaluation mode
    # ---------------------------------------------------------

    def train(self, mode: bool = True):
        self.training = bool(mode)

        for module in self._modules.values():
            module.train(mode)

        return self

    def eval(self):
        return self.train(False)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented."
        )

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self):
        if not self._modules:
            return f"{self.__class__.__name__}()"

        children = []

        for name, module in self._modules.items():
            children.append(
                f"({name}): {module}"
            )

        return (
            f"{self.__class__.__name__}(\n"
            + "\n".join(children)
            + "\n)"
        )
