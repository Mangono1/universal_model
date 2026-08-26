"""
Basic Universal Model Framework example.
"""

from universal_model import ModelConfig


def main():
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
    )

    print("Universal Model Framework")
    print("=========================")
    print("Version : 0.1.0")
    print("Config  :", config.to_dict())


if __name__ == "__main__":
    main()
