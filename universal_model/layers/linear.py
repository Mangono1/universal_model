"""
Linear layer for Universal Model Framework.

The tensor engine is CPUTorch.
"""

from ..core.module import Module
from ..core.parameter import Parameter
import cputorch


class Linear(Module):
    """
    Fully connected linear layer.

    Computes:

        y = x @ W + b

    CPUTorch 0.5.7 expects add_bias_2d()
    bias to be a 2D tensor, so bias uses shape:

        (1, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        if in_features <= 0:
            raise ValueError(
                "in_features must be greater than zero."
            )

        if out_features <= 0:
            raise ValueError(
                "out_features must be greater than zero."
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Temporary deterministic initialization.
        # Proper initialization will be added later.
        weight_values = [
            0.01
        ] * (self.in_features * self.out_features)

        weight_tensor = cputorch.Tensor(
            weight_values,
            shape=(self.in_features, self.out_features),
            requires_grad=True,
        )

        self.weight = Parameter(weight_tensor)
        self.register_parameter(
            "weight",
            self.weight,
        )

        self.bias = None

        if bias:
            bias_values = [
                0.0
            ] * self.out_features

            bias_tensor = cputorch.Tensor(
                bias_values,
                shape=(1, self.out_features),
                requires_grad=True,
            )

            self.bias = Parameter(bias_tensor)
            self.register_parameter(
                "bias",
                self.bias,
            )

    def forward(self, x):
        """
        Apply the linear transformation.

        Input:
            2D CPUTorch Tensor
            shape = (batch, in_features)

        Output:
            2D CPUTorch Tensor
            shape = (batch, out_features)
        """

        if not isinstance(x, cputorch.Tensor):
            raise TypeError(
                "Linear input must be a cputorch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "Linear currently expects a 2D tensor "
                "with shape (batch, in_features)."
            )

        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input with {self.in_features} features, "
                f"got {x.shape[1]}."
            )

        output = x.matmul(
            self.weight.tensor
        )

        if self.bias is not None:
            output = output.add_bias_2d(
                self.bias.tensor
            )

        return output

    def __repr__(self):
        return (
            f"Linear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f")"
        )
