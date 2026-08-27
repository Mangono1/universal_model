"""
Embedding layer for Universal Model Framework.
"""

from ..core.module import Module
from ..core.parameter import Parameter
from ..core.tensor_ops import TensorOps


class Embedding(Module):
    """
    Trainable embedding lookup table.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        num_embeddings = int(num_embeddings)
        embedding_dim = int(embedding_dim)

        if num_embeddings <= 0:
            raise ValueError(
                "num_embeddings must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            TensorOps.zeros(
                (num_embeddings, embedding_dim),
                requires_grad=True,
            )
        )

        self.register_parameter(
            "weight",
            self.weight,
        )

    def forward(self, indices):
        if isinstance(indices, int):
            indices = [indices]

        elif isinstance(indices, (list, tuple)):
            indices = list(indices)

        else:
            indices = TensorOps.ensure_tensor(
                indices,
                "indices",
            )

            if indices.ndim != 1:
                raise ValueError(
                    "indices tensor must be 1-dimensional."
                )

            indices = [
                int(indices.item(i))
                for i in range(indices.size)
            ]

        normalized = []

        for index in indices:
            if isinstance(index, bool):
                raise TypeError(
                    "embedding indices must be integers."
                )

            try:
                numeric_index = int(index)
            except (TypeError, ValueError):
                raise TypeError(
                    "embedding indices must be integers."
                )

            if numeric_index != index:
                raise TypeError(
                    "embedding indices must be integers."
                )

            if (
                numeric_index < 0
                or numeric_index >= self.num_embeddings
            ):
                raise IndexError(
                    f"Embedding index {numeric_index} is out of range. "
                    f"Expected 0 <= index < {self.num_embeddings}."
                )

            normalized.append(numeric_index)

        output = TensorOps.zeros(
            (len(normalized), self.embedding_dim),
            requires_grad=False,
        )

        weight_data = self.weight.tensor.data

        for row, index in enumerate(normalized):
            for column in range(self.embedding_dim):
                output.set_item(
                    row * self.embedding_dim + column,
                    weight_data[
                        index * self.embedding_dim + column
                    ],
                )

        return output

    def parameter_count(self):
        return (
            self.num_embeddings
            * self.embedding_dim
        )

    def __repr__(self):
        return (
            f"Embedding("
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}"
            f")"
        )


__all__ = [
    "Embedding",
]
