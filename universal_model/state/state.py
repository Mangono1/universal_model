"""
Model state and checkpoint serialization.

Universal Model Framework.

CPU-first, dependency-free serialization layer.
"""

import json
from pathlib import Path
from typing import Any, Dict

import cputorch


FORMAT_NAME = "universal_model"
FORMAT_VERSION = 1


class StateDict:
    """
    Lightweight container for model state.

    The state stores parameter values independently from
    the live model tensors.
    """

    def __init__(self, values=None):
        self.values = dict(values or {})

    def keys(self):
        return self.values.keys()

    def items(self):
        return self.values.items()

    def values_list(self):
        return self.values.values()

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __contains__(self, key):
        return key in self.values

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return (
            f"StateDict("
            f"parameters={len(self.values)}"
            f")"
        )


def _tensor_to_state(tensor):
    """
    Convert a CPUTorch tensor into a plain Python state object.
    """

    if tensor is None:
        raise ValueError(
            "Cannot serialize a None tensor."
        )

    shape = getattr(tensor, "shape", None)

    if shape is None:
        raise ValueError(
            "Tensor does not expose a shape."
        )

    data = getattr(tensor, "data", None)

    if data is None:
        raise ValueError(
            "Tensor does not expose data."
        )

    return {
        "shape": [
            int(value)
            for value in shape
        ],
        "data": [
            float(value)
            for value in data
        ],
        "requires_grad": bool(
            getattr(
                tensor,
                "requires_grad",
                True,
            )
        ),
    }


def _state_to_tensor(state):
    """
    Reconstruct a CPUTorch tensor from serialized state.
    """

    if not isinstance(state, dict):
        raise TypeError(
            "Tensor state must be a dictionary."
        )

    shape = state.get("shape")
    data = state.get("data")

    if shape is None:
        raise ValueError(
            "Tensor state is missing 'shape'."
        )

    if data is None:
        raise ValueError(
            "Tensor state is missing 'data'."
        )

    shape = tuple(
        int(value)
        for value in shape
    )

    expected_size = 1

    for dimension in shape:
        expected_size *= dimension

    if len(data) != expected_size:
        raise ValueError(
            "Tensor state size mismatch: "
            f"shape={shape}, "
            f"data={len(data)}."
        )

    return cputorch.Tensor(
        [
            float(value)
            for value in data
        ],
        shape=shape,
        requires_grad=bool(
            state.get(
                "requires_grad",
                True,
            )
        ),
    )


def state_dict(model):
    """
    Create an independent StateDict from a model.

    The returned values are copies, so modifying the live model
    after this call does not modify the saved state.
    """

    if model is None:
        raise ValueError(
            "model must not be None."
        )

    if not hasattr(model, "named_parameters"):
        raise TypeError(
            "Object must provide named_parameters()."
        )

    values = {}

    for name, parameter in model.named_parameters():
        tensor = getattr(
            parameter,
            "tensor",
            parameter,
        )

        values[name] = _tensor_to_state(
            tensor
        )

    return StateDict(values)


def load_state_dict(
    model,
    state,
    strict=True,
):
    """
    Load a StateDict into an existing model.

    Existing Parameter and Tensor objects are preserved.
    """

    if model is None:
        raise ValueError(
            "model must not be None."
        )

    if isinstance(state, StateDict):
        values = state.values
    elif isinstance(state, dict):
        values = state
    else:
        raise TypeError(
            "state must be a StateDict or dict."
        )

    current = dict(
        model.named_parameters()
    )

    incoming_names = set(
        values.keys()
    )

    current_names = set(
        current.keys()
    )

    missing = sorted(
        current_names - incoming_names
    )

    unexpected = sorted(
        incoming_names - current_names
    )

    if strict and (
        missing or unexpected
    ):
        parts = []

        if missing:
            parts.append(
                f"missing keys: {missing}"
            )

        if unexpected:
            parts.append(
                f"unexpected keys: {unexpected}"
            )

        raise ValueError(
            "State dictionary mismatch: "
            + "; ".join(parts)
        )

    loaded = 0

    for name, serialized in values.items():
        if name not in current:
            continue

        parameter = current[name]
        tensor = getattr(
            parameter,
            "tensor",
            None,
        )

        incoming = _state_to_tensor(
            serialized
        )

        existing_shape = getattr(
            tensor,
            "shape",
            None,
        )

        if (
            existing_shape is not None
            and tuple(existing_shape)
            != tuple(incoming.shape)
        ):
            raise ValueError(
                f"Shape mismatch for '{name}': "
                f"expected {existing_shape}, "
                f"got {incoming.shape}."
            )

        if tensor is None:
            parameter.tensor = incoming
            loaded += 1
            continue

        for index, value in enumerate(
            incoming.data
        ):
            tensor.set_item(
                index,
                value,
            )

        loaded += 1

    return loaded


def save_model(model, path):
    """
    Save model parameters to a JSON checkpoint.

    Parameters
    ----------
    model:
        Universal Model Module.

    path:
        Destination checkpoint path.
    """

    state = state_dict(model)

    checkpoint = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "type": model.__class__.__name__,
        "parameters": state.values,
    }

    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            checkpoint,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return destination


def load_model(
    model,
    path,
    strict=True,
):
    """
    Load model parameters from a checkpoint.

    The model architecture must already exist.
    """

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: "
            f"{source}"
        )

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        checkpoint = json.load(
            handle
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Invalid model checkpoint."
        )

    if checkpoint.get("format") != FORMAT_NAME:
        raise ValueError(
            "Invalid checkpoint format: "
            f"{checkpoint.get('format')}"
        )

    version = checkpoint.get(
        "version"
    )

    if version != FORMAT_VERSION:
        raise ValueError(
            "Unsupported checkpoint version: "
            f"{version}"
        )

    parameters = checkpoint.get(
        "parameters"
    )

    if not isinstance(
        parameters,
        dict,
    ):
        raise ValueError(
            "Checkpoint does not contain "
            "a valid parameter dictionary."
        )

    return load_state_dict(
        model,
        StateDict(parameters),
        strict=strict,
    )


__all__ = [
    "StateDict",
    "state_dict",
    "load_state_dict",
    "save_model",
    "load_model",
]
