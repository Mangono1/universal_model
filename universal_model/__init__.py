"""
Universal Model Framework.

A lightweight CPU-first neural network framework.
"""

from .core.module import Module
from .core.parameter import Parameter
from .core.tensor_ops import TensorOps

from .layers.linear import Linear
from .layers.activation import ReLU
from .layers.sequential import Sequential

from .losses.mse import MSELoss

from .optim.optimizer import Optimizer
from .optim.sgd import SGD

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

from .training.history import TrainingHistory
from .training.trainer import Trainer

from .evaluation.metrics import (
    mse,
    mae,
    rmse,
)

from .evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
)

from .state import (
    StateDict,
    state_dict,
    load_state_dict,
    save_model,
    load_model,
)

__all__ = [
    "Module",
    "Parameter",
    "TensorOps",

    "Linear",
    "ReLU",
    "Sequential",

    "MSELoss",

    "Optimizer",
    "SGD",

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

    "TrainingHistory",
    "Trainer",

    "mse",
    "mae",
    "rmse",

    "Evaluator",
    "EvaluationResult",

    "StateDict",
    "state_dict",
    "load_state_dict",
    "save_model",
    "load_model",
]