import pytest
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import BinaryCalibrationError

from pytorch_utils.metrics import (
    WeightedBinaryCalibrationError,
    WeightedMeanAbsoluteError,
    WeightedMeanSquaredError,
)


@pytest.fixture
def regression_predictions():
    return torch.tensor([2, 4, 3, 8, 1])


@pytest.fixture
def regression_targets():
    return torch.tensor([1, 2, 3, 6, 1])


@pytest.fixture
def callibration_predictions():
    return torch.tensor([1e-4, 0.2, 0.7, 0.7, 0.2, 1])


@pytest.fixture
def callibration_targets():
    return torch.tensor([0, 0, 0, 1, 1, 1])


def check_perfect_predictions(weighted_metric, preds, targets):
    torch.testing.assert_close(weighted_metric(preds, targets), torch.tensor(0.0))

    torch.testing.assert_close(
        weighted_metric(preds, targets, torch.ones(len(preds))), torch.tensor(0.0)
    )


def check_equality_mask(weighted_metric, preds, targets):
    torch.testing.assert_close(
        weighted_metric(preds, targets, 17 * (targets == preds)), torch.tensor(0.0)
    )


def compare_metrics(metric, weighted_metric, preds, targets):
    torch.testing.assert_close(metric(preds, targets), weighted_metric(preds, targets))

    torch.testing.assert_close(
        metric(preds, targets), weighted_metric(preds, targets, torch.ones(len(preds)))
    )

    weights = (targets != preds) * 1
    torch.testing.assert_close(
        len(preds) / weights.sum() * metric(preds, targets),
        weighted_metric(preds, targets, weights),
    )

    n = len(preds) // 2
    weights = torch.tensor([3] * n + [1] * (len(preds) - n))
    torch.testing.assert_close(
        metric(
            torch.cat((preds, preds[:n], preds[:n])),
            torch.cat((targets, targets[:n], targets[:n])),
        ),
        weighted_metric(preds, targets, weights),
    )


def compare_regression_metrics(metric, weighted_metric, preds, targets):
    compare_metrics(metric, weighted_metric, preds, targets)

    weights = (targets != preds) * 2 + (targets == preds) * 1
    torch.testing.assert_close(
        2 * len(preds) / weights.sum() * metric(preds, targets),
        weighted_metric(preds, targets, weights),
    )


def test_weighted_mse(regression_predictions, regression_targets):
    mse = MeanSquaredError()
    weighted_mse = WeightedMeanSquaredError()

    check_perfect_predictions(weighted_mse, regression_predictions, regression_predictions)
    check_perfect_predictions(weighted_mse, regression_targets, regression_targets)
    check_equality_mask(weighted_mse, regression_predictions, regression_targets)
    compare_regression_metrics(mse, weighted_mse, regression_predictions, regression_targets)


def test_weighted_mae(regression_predictions, regression_targets):
    mae = MeanAbsoluteError()
    weighted_mae = WeightedMeanAbsoluteError()

    check_perfect_predictions(weighted_mae, regression_predictions, regression_predictions)
    check_perfect_predictions(weighted_mae, regression_targets, regression_targets)
    check_equality_mask(weighted_mae, regression_predictions, regression_targets)
    compare_regression_metrics(mae, weighted_mae, regression_predictions, regression_targets)


def test_weighted_callibration(callibration_predictions, callibration_targets):
    callibration = BinaryCalibrationError()
    weighted_callibration = WeightedBinaryCalibrationError()

    zero_labels = callibration_targets == 0
    check_perfect_predictions(
        weighted_callibration,
        callibration_targets.float() + 1e-6 * zero_labels,
        callibration_targets,
    )
    check_equality_mask(weighted_callibration, callibration_predictions, callibration_targets)

    print(weighted_callibration(callibration_predictions, callibration_targets))
    compare_metrics(
        callibration,
        weighted_callibration,
        callibration_predictions,
        callibration_targets,
    )

    weights = (1 - callibration_predictions) * (
        1 - callibration_targets
    ) + callibration_predictions * callibration_targets
    torch.testing.assert_close(
        weighted_callibration(callibration_predictions, callibration_targets, weights),
        torch.tensor(0.0),
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        weighted_callibration(
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
            torch.tensor([0, 0, 0, 0, 1]),
            torch.tensor([0, 0, 0, 4, 1]),
        ),
        torch.tensor(0.0),
    )
