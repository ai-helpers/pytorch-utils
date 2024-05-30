from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pytorch_utils.pandas.utils import ListDataFrameRows
from pytorch_utils.dataset_configurations import (
    AugmentedBernoulliDatasetConfig,
    DataAugmentationConfig,
    DataSplitConfig,
)
from pytorch_utils.datasets import (
    AugmentedBernoulliDataset,
    MLStage,
)


@pytest.fixture(scope="session")
def test_dataset():
    return pd.read_csv("data/expected/example_sample.csv")


def test_instanciation(test_dataset: pd.DataFrame):
    scaling_factors = np.arange(1, 3, 0.2)
    dataset = AugmentedBernoulliDataset(
        data=ListDataFrameRows.from_pandas_df(test_dataset),
        data_augmentation_scaling_factors=scaling_factors,
        is_success=True,
        augmented_col="percent",
    )
    assert len(dataset) == len(test_dataset) * len(scaling_factors)
    assert len(dataset.raw_feature_names) == len(test_dataset.columns)
    assert len(dataset.transformed_feature_names) == len(test_dataset.columns)


def test_immutability(test_dataset):
    dataset = AugmentedBernoulliDataset(
        data=ListDataFrameRows.from_pandas_df(test_dataset),
        data_augmentation_scaling_factors=np.arange(1, 3, 0.2),
        is_success=True,
        augmented_col="percent",
    )
    with pytest.raises(AttributeError):
        dataset.data = []

    with pytest.raises(AttributeError):
        dataset.data_augmentation_scaling_factors = np.arange(1, 3, 0.1)

    with pytest.raises(AttributeError):
        dataset.augmented_df_indices = dataset._build_augmentation_indices()

    dataset_copy = replace(dataset, data_augmentation_scaling_factors=np.arange(1, 3, 0.1))
    assert dataset_copy.data_augmentation_scaling_factors == pytest.approx(np.arange(1, 3, 0.1))


def test_pandas_equivalence(test_dataset):
    config = AugmentedBernoulliDatasetConfig(
        data=test_dataset,
        is_success=True,
        data_augmentation_config=DataAugmentationConfig(
            augmented_col="percent",
            scaling_factors=np.arange(1, 3, 0.1),
            min_value=0.0,
            max_value=1.0,
        ),
        split_config=DataSplitConfig(),
    )

    dataset = AugmentedBernoulliDataset.from_config(
        config=config,
        ml_stage=MLStage.fit,
        fitted_preprocessing_pipeline=None,
    )

    augmented_training_data = config.augmented_training_data
    assert len(dataset) == len(augmented_training_data)

    df_from_dataset = ListDataFrameRows.from_list_rows(
        [dataset[i] for i in range(len(dataset))]
    ).dataframe

    assert df_from_dataset[dataset.label_col].all()

    pd.testing.assert_frame_equal(
        df_from_dataset[config.augmented_training_data.columns].reset_index(drop=True),
        config.augmented_training_data,
    )
