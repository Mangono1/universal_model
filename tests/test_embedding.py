import cputorch

from universal_model import Embedding


def test_embedding_shape():
    embedding = Embedding(
        num_embeddings=10,
        embedding_dim=4,
    )

    output = embedding([1, 3, 7])

    assert output.shape == (3, 4)


def test_embedding_parameter_shape():
    embedding = Embedding(
        num_embeddings=100,
        embedding_dim=16,
    )

    assert embedding.weight.shape == (100, 16)
    assert embedding.parameter_count() == 1600


def test_embedding_single_token():
    embedding = Embedding(
        num_embeddings=8,
        embedding_dim=4,
    )

    output = embedding(2)

    assert output.shape == (1, 4)


def test_embedding_tensor_input():
    embedding = Embedding(
        num_embeddings=8,
        embedding_dim=4,
    )

    ids = cputorch.Tensor(
        [1.0, 2.0, 3.0],
        shape=(3,),
    )

    output = embedding(ids)

    assert output.shape == (3, 4)


def test_embedding_invalid_index():
    embedding = Embedding(
        num_embeddings=5,
        embedding_dim=4,
    )

    try:
        embedding([0, 5])
        assert False
    except IndexError:
        assert True


def test_embedding_repr():
    embedding = Embedding(
        num_embeddings=10,
        embedding_dim=8,
    )

    assert repr(embedding) == (
        "Embedding(num_embeddings=10, embedding_dim=8)"
    )
