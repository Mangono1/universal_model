"""
Sequential container for Universal Model Framework.
"""

from typing import Iterable

from ..core.module import Module


class Sequential(Module):
    """
    Apply child modules sequentially.

    Example:

        Sequential(
            Linear(2, 8),
            ReLU(),
            Linear(8, 4),
        )
    """

    def __init__(self, *modules: Module):
        super().__init__()

        for index, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(
                    "Sequential modules must inherit from Module."
                )

            self.add_module(str(index), module)

    def forward(self, x):
        output = x

        for module in self._modules.values():
            output = module(output)

        return output

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, index):
        modules = list(self._modules.values())
        return modules[index]

    def __repr__(self):
        if not self._modules:
            return "Sequential()"

        lines = ["Sequential("]

        for index, module in enumerate(self._modules.values()):
            lines.append(
                f"  ({index}): {module}"
            )

        lines.append(")")

        return "\n".join(lines)
