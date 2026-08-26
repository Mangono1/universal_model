"""
Core module abstraction for Universal Model Framework.
"""

from typing import Dict, Iterator


class Module:
    """
    Base class for all trainable Universal Model components.

    Modules can contain parameters and child modules.
    """

    def __init__(self):
        self._parameters: Dict[str, object] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training = True

    def register_parameter(self, name: str, parameter: object) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter name must be a non-empty string.")

        self._parameters[name] = parameter

    def add_module(self, name: str, module: "Module") -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Module name must be a non-empty string.")

        if not isinstance(module, Module):
            raise TypeError("Child module must inherit from Module.")

        self._modules[name] = module

    def parameters(self) -> Iterator[object]:
        for parameter in self._parameters.values():
            yield parameter

        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = ""):
        for name, parameter in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, parameter

        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(module_prefix)

    def train(self, mode: bool = True):
        self.training = mode

        for module in self._modules.values():
            module.train(mode)

        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented."
        )
