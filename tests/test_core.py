from universal_model import ModelConfig, Module, Parameter


def test_module():
    module = Module()
    assert module.training is True


def test_parameter():
    parameter = Parameter("tensor")
    assert parameter.requires_grad is True
    assert parameter.tensor == "tensor"


def test_config():
    config = ModelConfig()

    assert config.vocab_size == 32000
    assert config.hidden_size == 512
    assert config.num_layers == 6
    assert config.num_heads == 8
