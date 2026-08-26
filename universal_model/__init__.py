"""
Universal Model Framework.

A universal neural-model framework built on top of CPUTorch.
"""

__version__ = "0.1.0"
__author__ = "Frandika Imam Arifin"


from .core.module import Module
from .core.parameter import Parameter
from .core.tensor_ops import TensorOps

from .models.config import ModelConfig
from .models.base import BaseModel

from .layers.linear import Linear
from .layers.activation import ReLU
from .layers.sequential import Sequential

from .init.initializers import (
    zeros,
    ones,
    constant,
    uniform,
    normal,
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    initialize,
)

from .losses.mse import MSELoss


__all__ = [
    "Module",
    "Parameter",
    "TensorOps",

    "ModelConfig",
    "BaseModel",

    "Linear",
    "ReLU",
    "Sequential",

    "zeros",
    "ones",
    "constant",
    "uniform",
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "initialize",

    "MSELoss",
]

from .optim.optimizer import Optimizer
from .optim.sgd import SGD


__all__.extend(["Optimizer", "SGD"])

