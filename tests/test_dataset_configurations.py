import pandas as pd
import pytest

from pytorch_utils.dataset_configurations import (
    DataSplitConfig,
)


@pytest.fixture()
def test_dataset():
    return pd.read_csv("data/expected/example_sample.csv")


def test_train_valid_test_split(test_dataset: pd.DataFrame):
    # Define configuration
    config = DataSplitConfig(
        training_proportion=0.64,
        validation_proportion=0.16,
        test_proportion=0.2,
        random_seed=42,
    )

    # Apply split
    train, validation, test = config.train_valid_test_split(test_dataset)

    # Check if union is consistant
    assert test_dataset.index.equals(train.union(validation).union(test))

    # Check if proportions are valid
    assert pytest.approx(len(train) / len(test_dataset), abs=1 / (len(test_dataset) - 1)) == 0.64
    assert (
        pytest.approx(len(validation) / len(test_dataset), abs=1 / (len(test_dataset) - 1)) == 0.16
    )
    assert pytest.approx(len(test) / len(test_dataset), abs=1 / (len(test_dataset) - 1)) == 0.2
