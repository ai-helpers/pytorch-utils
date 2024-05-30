import pytest
import torch

from pytorch_utils.modules import (
    BatchNorm1dNonNeg,
    BiLinearSemiNonNeg,
    LinearNonNeg,
    Partitioned,
    ShiftedEmbedding,
)


def test_non_neg_linear():
    n = 500
    inputs = torch.rand(1, n, requires_grad=True)

    # Check that for a traditional randomly initialized linear
    # layer the gradient of the outputs w.r.t. to the inputs
    # is NOT non-negative
    lin_layer = torch.nn.Linear(n, n)
    lin_layer(inputs).sum().backward()
    assert not torch.all(inputs.grad >= 0.0)

    # Check that for a randomly initialized LinearNonNeg
    # layer the gradient of the outputs w.r.t. to the inputs
    # is always non-negative
    inputs.grad = None
    non_neg_lin_layer = LinearNonNeg(n, n)
    non_neg_lin_layer(inputs).sum().backward()
    assert torch.all(inputs.grad >= 0.0)


def test_non_neg_batch_norm():
    n = 5
    non_neg_batch_norm = BatchNorm1dNonNeg(n)
    non_neg_batch_norm.train(False)
    optimizer = torch.optim.SGD(params=[non_neg_batch_norm.weight], lr=1.0)

    inputs = torch.tensor([[1 for _ in range(n)], [2 for _ in range(n)], [3.0 for _ in range(n)]])

    for _ in range(100):
        optimizer.zero_grad()
        loss = non_neg_batch_norm(inputs).sum()
        loss.backward()
        optimizer.step()

    # Test that even after trying to minimize the weights
    # the outputs of batch norm is still non-negative
    assert torch.all(non_neg_batch_norm(inputs) >= 0.0)

    # Also chack that the gradient with respect to the inputs
    # is also non-negative
    inputs.requires_grad = True
    non_neg_batch_norm(inputs).sum().backward()
    assert torch.all(inputs.grad >= 0.0)


def test_bilinear_semi_non_neg():
    bilinear_semi_non_neg_layer = BiLinearSemiNonNeg(
        in_features_non_neg=2,
        in_features_others=3,
        out_features_non_neg=4,
        out_features_others=5,
        non_neg_inputs_name="non_neg_inputs",
        other_inputs_name="other_inputs",
    )

    inputs = {
        "non_neg_inputs": torch.tensor([[1.0, 2]]),
        "other_inputs": torch.tensor([[1.0, 2, 3]]),
    }

    bilinear_semi_non_neg_layer(inputs)

    # TODO: check gradient


def test_partitioned():
    lin_layers = [
        torch.nn.Linear(2, 3),
        torch.nn.Linear(1, 2),
        torch.nn.Linear(4, 1),
    ]

    partitioned_layer = Partitioned(
        feat1=lin_layers[0],
        feat2=lin_layers[1],
        feat3=lin_layers[2],
    )
    partitioned_layer.train(False)

    inputs = {
        "feat1": torch.tensor([[1.0, 2]]),
        "feat2": torch.tensor(
            [
                [
                    1.0,
                ]
            ]
        ),
        "feat3": torch.tensor([[1.0, 2, 3, 4]]),
    }
    outputs = partitioned_layer(inputs)

    # Check output format
    assert len(outputs) == 3
    for f in inputs:
        assert f in outputs

    # Check eval mode
    for lin_layer in lin_layers:
        assert not lin_layer.training

    # Check output values
    for f, lin_layer in zip(outputs, lin_layers):
        print(f)
        torch.testing.assert_close(outputs[f], lin_layer(inputs[f]))


def test_shifted_embedding():
    embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=5)
    shifted_embedding = ShiftedEmbedding(num_embeddings=10, embedding_dim=5)

    embedding(torch.tensor([i for i in range(10)]))

    with pytest.raises(IndexError):
        embedding(torch.tensor([10]))

    with pytest.raises(IndexError):
        embedding(torch.tensor([-1]))

    shifted_embedding(torch.tensor([i for i in range(-1, 9)]))

    with pytest.raises(IndexError):
        shifted_embedding(torch.tensor([-2]))

    with pytest.raises(IndexError):
        shifted_embedding(torch.tensor([9]))
