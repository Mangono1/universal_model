"""
Universal Model Framework.

A universal neural-model framework built on top of CPUTorch.
"""

__version__ = "0.1.0"
__author__ = "Frandika Imam Arifin"

from .core.module import Module
from .core.parameter import Parameter
from .models.config import ModelConfig
from .models.base import BaseModel

__all__ = [
    "Module",
    "Parameter",
    "ModelConfig",
    "BaseModel",
]
