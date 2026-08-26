"""
Parameter initialization utilities for Universal Model Framework.
"""

import math
import random

import cputorch


def _validate_tensor(tensor):
    if not isinstance(tensor, cputorch.Tensor):
        raise TypeError("Expected a cputorch.Tensor.")


def _validate_shape(tensor):
    _validate_tensor(tensor)

    if not tensor.shape:
        raise ValueError("Tensor must have a valid shape.")

    return tuple(int(x) for x in tensor.shape)


def _set_flat_values(tensor, values):
    _validate_tensor(tensor)

    if len(values) != tensor.size:
        raise ValueError(
            f"Expected {tensor.size} values, got {len(values)}."
        )

    for index, value in enumerate(values):
        tensor.set_item(index, float(value))

    return tensor


def _fan_in_out(tensor):
    shape = _validate_shape(tensor)

    if len(shape) < 2:
        fan_in = shape[0]
        fan_out = shape[0]
    else:
        receptive_field = 1

        for dimension in shape[2:]:
            receptive_field *= dimension

        fan_in = shape[0] * receptive_field
        fan_out = shape[1] * receptive_field

    return fan_in, fan_out


def zeros(tensor):
    """Initialize tensor with zeros."""
    return _set_flat_values(
        tensor,
        [0.0] * tensor.size,
    )


def ones(tensor):
    """Initialize tensor with ones."""
    return _set_flat_values(
        tensor,
        [1.0] * tensor.size,
    )


def constant(tensor, value):
    """Initialize tensor with a constant value."""
    return _set_flat_values(
        tensor,
        [float(value)] * tensor.size,
    )


def uniform(tensor, low=0.0, high=1.0, seed=None):
    """Initialize tensor using a uniform distribution."""
    low = float(low)
    high = float(high)

    if low > high:
        raise ValueError("low must be <= high.")

    rng = random.Random(seed)

    values = [
        rng.uniform(low, high)
        for _ in range(tensor.size)
    ]

    return _set_flat_values(tensor, values)


def normal(tensor, mean=0.0, std=1.0, seed=None):
    """Initialize tensor using a normal distribution."""
    mean = float(mean)
    std = float(std)

    if std < 0.0:
        raise ValueError("std must be >= 0.")

    rng = random.Random(seed)

    values = [
        rng.gauss(mean, std)
        for _ in range(tensor.size)
    ]

    return _set_flat_values(tensor, values)


def xavier_uniform(tensor, gain=1.0, seed=None):
    """
    Xavier/Glorot uniform initialization.
    """
    gain = float(gain)

    fan_in, fan_out = _fan_in_out(tensor)

    if fan_in + fan_out <= 0:
        raise ValueError(
            "fan_in + fan_out must be greater than zero."
        )

    limit = gain * math.sqrt(
        6.0 / float(fan_in + fan_out)
    )

    return uniform(
        tensor,
        low=-limit,
        high=limit,
        seed=seed,
    )


def xavier_normal(tensor, gain=1.0, seed=None):
    """
    Xavier/Glorot normal initialization.
    """
    gain = float(gain)

    fan_in, fan_out = _fan_in_out(tensor)

    if fan_in + fan_out <= 0:
        raise ValueError(
            "fan_in + fan_out must be greater than zero."
        )

    std = gain * math.sqrt(
        2.0 / float(fan_in + fan_out)
    )

    return normal(
        tensor,
        mean=0.0,
        std=std,
        seed=seed,
    )


def kaiming_uniform(tensor, a=0.0, seed=None):
    """
    Kaiming/He uniform initialization.
    """
    a = float(a)

    fan_in, _ = _fan_in_out(tensor)

    if fan_in <= 0:
        raise ValueError(
            "fan_in must be greater than zero."
        )

    bound = math.sqrt(
        6.0 / ((1.0 + a * a) * float(fan_in))
    )

    return uniform(
        tensor,
        low=-bound,
        high=bound,
        seed=seed,
    )


def kaiming_normal(tensor, a=0.0, seed=None):
    """
    Kaiming/He normal initialization.
    """
    a = float(a)

    fan_in, _ = _fan_in_out(tensor)

    if fan_in <= 0:
        raise ValueError(
            "fan_in must be greater than zero."
        )

    std = math.sqrt(
        2.0 / ((1.0 + a * a) * float(fan_in))
    )

    return normal(
        tensor,
        mean=0.0,
        std=std,
        seed=seed,
    )


def initialize(tensor, method="xavier_uniform", **kwargs):
    """
    Generic initialization dispatcher.
    """
    methods = {
        "zeros": zeros,
        "ones": ones,
        "constant": constant,
        "uniform": uniform,
        "normal": normal,
        "xavier_uniform": xavier_uniform,
        "xavier_normal": xavier_normal,
        "kaiming_uniform": kaiming_uniform,
        "kaiming_normal": kaiming_normal,
    }

    if method not in methods:
        available = ", ".join(sorted(methods))

        raise ValueError(
            f"Unknown initialization method '{method}'. "
            f"Available methods: {available}"
        )

    return methods[method](tensor, **kwargs)


__all__ = [
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
]
