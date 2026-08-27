import cputorch

from universal_model import MSELoss


def tensor(values):
    return cputorch.Tensor(
        values,
        shape=(len(values),),
    )


def test_mse_mean():
    prediction = tensor([1.0, 2.0, 3.0])
    target = tensor([0.0, 2.0, 5.0])

    loss = MSELoss()(prediction, target)

    assert loss.shape == (1,)
    assert abs(loss.item(0) - (5.0 / 3.0)) < 1e-6


def test_mse_sum():
    prediction = tensor([1.0, 2.0, 3.0])
    target = tensor([0.0, 2.0, 5.0])

    loss = MSELoss(reduction="sum")(prediction, target)

    assert loss.shape == (1,)
    assert abs(loss.item(0) - 5.0) < 1e-6


def test_mse_invalid_reduction():
    try:
        MSELoss(reduction="invalid")
        assert False
    except ValueError:
        assert True


def test_mse_shape_mismatch():
    prediction = tensor([1.0, 2.0, 3.0])
    target = tensor([1.0, 2.0])

    try:
        MSELoss()(prediction, target)
        assert False
    except ValueError:
        assert True


def test_mse_requires_tensor():
    try:
        MSELoss()([1.0, 2.0], [1.0, 2.0])
        assert False
    except TypeError:
        assert True
