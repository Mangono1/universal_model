"""
Universal model configuration.
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ModelConfig:
    """
    General configuration shared by Universal Model architectures.

    The configuration intentionally does not contain any domain-specific
    information. Users are free to build agriculture, medical, science,
    education, or other domain models.
    """

    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 2048
    max_sequence_length: int = 1024
    dropout: float = 0.0

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        if self.intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be greater than zero."
            )

        if self.max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be greater than zero."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ModelConfig":
        return cls(**data)
