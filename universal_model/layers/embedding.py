"""
Embedding layer for Universal Model Framework.
"""

import cputorch

from ..core.module import Module
from ..core.parameter import Parameter


class Embedding(Module):
    """
    Learnable token embedding layer.

    Uses one-hot matrix multiplication because the current CPUTorch
    API does not expose an indexed gather operation.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be greater than zero.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        values = []

        for row in range(self.num_embeddings):
            for column in range(self.embedding_dim):
                values.append(
                    0.01 * ((row + column) % 10 + 1)
                )

        tensor = cputorch.Tensor(
            values,
            shape=(self.num_embeddings, self.embedding_dim),
            requires_grad=True,
        )

        self.weight = Parameter(tensor)

        self.register_parameter(
            "weight",
            self.weight,
        )

    def _normalize_indices(self, indices):
        if isinstance(indices, cputorch.Tensor):
            if indices.ndim != 1:
                raise ValueError(
                    "Embedding currently expects a 1D tensor of token IDs."
                )

            values = indices.data

        elif isinstance(indices, int):
            values = [indices]

        else:
            try:
                values = list(indices)
            except TypeError as exc:
                raise TypeError(
                    "Embedding indices must be an int, iterable of ints, "
                    "or a 1D cputorch.Tensor."
                ) from exc

        result = []

        for index in values:
            if isinstance(index, bool):
                raise TypeError(
                    "Embedding indices must be integers."
                )

            if not isinstance(index, int):
                if isinstance(index, float) and index.is_integer():
                    index = int(index)
                else:
                    raise TypeError(
                        "Embedding indices must be integers."
                    )

            if index < 0 or index >= self.num_embeddings:
                raise IndexError(
                    f"Embedding index {index} is out of range "
                    f"for {self.num_embeddings} embeddings."
                )

            result.append(int(index))

        return result

    def forward(self, indices):
        """
        Input:
            1D sequence of integer token IDs.

        Output:
            2D CPUTorch Tensor:
            (number_of_tokens, embedding_dim)
        """

        token_ids = self._normalize_indices(indices)

        if not token_ids:
            return cputorch.Tensor(
                [],
                shape=(0, self.embedding_dim),
                requires_grad=False,
            )

        one_hot = [
            0.0
        ] * (
            len(token_ids) * self.num_embeddings
        )

        for row, token_id in enumerate(token_ids):
            one_hot[
                row * self.num_embeddings + token_id
            ] = 1.0

        one_hot_tensor = cputorch.Tensor(
            one_hot,
            shape=(
                len(token_ids),
                self.num_embeddings,
            ),
            requires_grad=False,
        )

        return one_hot_tensor.matmul(
            self.weight.tensor
        )

    def __repr__(self):
        return (
            f"Embedding("
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}"
            f")"
        )
