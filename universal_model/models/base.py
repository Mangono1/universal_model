"""
Base model abstraction.
"""

from ..core.module import Module


class BaseModel(Module):
    """
    Base class for all Universal Model architectures.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def parameter_count(self) -> int:
        """
        Return the number of scalar parameters.

        Individual model implementations can override this when the
        underlying CPUTorch tensor representation exposes exact sizes.
        """
        total = 0

        for parameter in self.parameters():
            tensor = getattr(parameter, "tensor", parameter)

            shape = getattr(tensor, "shape", None)

            if shape is not None:
                try:
                    count = 1
                    for dimension in shape:
                        count *= int(dimension)
                    total += count
                except (TypeError, ValueError):
                    pass

        return total
