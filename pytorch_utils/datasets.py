from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset

from pytorch_utils.pandas.utils import DataFrameRow, ListDataFrameRows
from pytorch_utils.dataset_configurations import (
    AugmentedBernoulliDatasetConfig,
    DataAugmentationConfig,
)


class MLStage(Enum):
    fit = auto()
    validate = auto()
    test = auto()
    predict = auto()


@dataclass(frozen=True)
class AugmentedBernoulliDataset(Dataset):
    """
    Implements storage-efficient data augmentation of Bernoulli samples
    (binary outcomes: successful or not) as well as data transformations
    (e.g., scaling, encoding, ...).
    If `is_success` is set to None, only features are generated (labels are dropped).
    This is useful for prediction sets.
    """

    data: ListDataFrameRows
    is_success: Optional[
        bool
    ]  # if None then only features are generated, if True labels are set to 1, If False labels are set to 0
    augmented_col: str
    fitted_preprocessing_pipeline: Optional[Pipeline] = None
    data_augmentation_scaling_factors: np.ndarray = np.array([1.0])
    label_col: str = "success_labels"  # name of label column
    labels_dtype: np.dtype = np.dtype("int32")
    sample_weight_col: Optional[str] = None
    min_augmented_value: float = -float("inf")
    max_augmented_value: float = float("inf")

    def __post_init__(self) -> None:
        # Sort data_augmentation scaling factors
        # This ensures that two dataclasses that only differ by the ordering of
        # `data_augmentation_scaling_factors` are considered equal
        object.__setattr__(
            self,
            "data_augmentation_scaling_factors",
            sorted(self.data_augmentation_scaling_factors),
        )

        # Check attributes
        if self.augmented_col not in self.data.columns:
            raise ValueError("`augmented_col` should be a column of dataframe `data`")

        if self.fitted_preprocessing_pipeline:
            check_is_fitted(
                self.fitted_preprocessing_pipeline,
                msg="fitted_preprocessing_pipeline is not fitted",
            )

        if len(self.data_augmentation_scaling_factors) == 0:
            raise ValueError("`data_augmentation_scaling_factors` should not be an empty array")

        # Compute augmented_df_indices and augmented_df_lengths_cumsum based on data_augmentation_scaling_factors.
        object.__setattr__(self, "augmented_df_indices", self._build_augmented_df_indices())
        object.__setattr__(
            self,
            "augmented_df_lengths_cumsum",
            self._build_augmented_df_lengths_cumsum(),
        )
        object.__setattr__(
            self,
            "indices_rows_mapping",
            {row.index: i for i, row in enumerate(self.data)},
        )

        # Set pipeline ouput to pandas
        if self.fitted_preprocessing_pipeline:
            self.fitted_preprocessing_pipeline.set_output(transform="pandas")

    @classmethod
    def from_config(
        cls,
        config: AugmentedBernoulliDatasetConfig,
        ml_stage: Literal[MLStage.fit, MLStage.validate, MLStage.test],
        fitted_preprocessing_pipeline: Optional[Pipeline] = None,
        label_col: str = "success_labels",
        labels_dtype: np.dtype = np.dtype("int32"),
        sample_weight_col: Optional[str] = None,
    ) -> AugmentedBernoulliDataset:
        if ml_stage not in [MLStage.fit, MLStage.validate, MLStage.test]:
            raise ValueError(
                "ml_stage should be one of MLStage.fit, MLStage.validate or MLStage.test"
            )

        return cls(
            data=ListDataFrameRows.from_pandas_df(
                config.training_data
                if ml_stage is MLStage.fit
                else config.validation_data
                if ml_stage is MLStage.validate
                else config.test_data
                if ml_stage is MLStage.test
                else None
            ),
            is_success=config.is_success,
            augmented_col=config.augmented_col,
            fitted_preprocessing_pipeline=fitted_preprocessing_pipeline,
            data_augmentation_scaling_factors=config.data_augmentation_scaling_factors,
            label_col=label_col,
            labels_dtype=labels_dtype,
            sample_weight_col=sample_weight_col,
            min_augmented_value=config.data_augmentation_config.min_value,
            max_augmented_value=config.data_augmentation_config.max_value,
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.data.dataframe

    def clear_data(self) -> AugmentedBernoulliDataset:
        object.__setattr__(self, "data", self.data[:0])
        return self

    def _build_augmented_df_indices(self) -> List[pd.Index]:
        df = self.data.build_dataframe([self.augmented_col])
        return [
            df[
                DataAugmentationConfig.scaling_filter(
                    df,
                    scaling_factor,
                    self.augmented_col,
                    min_value=self.min_augmented_value,
                    max_value=self.max_augmented_value,
                )
            ].index
            for scaling_factor in self.data_augmentation_scaling_factors
        ]

    def _build_augmented_df_lengths_cumsum(self) -> np.ndarray:
        return np.cumsum([len(indices) for indices in getattr(self, "augmented_df_indices")])

    @property
    def raw_feature_names(self):
        return self.data.columns

    @property
    def transformed_feature_names(self):
        return (
            self.fitted_preprocessing_pipeline.get_feature_names_out()
            if self.fitted_preprocessing_pipeline
            else self.data.columns
        )

    def __getitem__(self, index) -> DataFrameRow:
        """
        Implicit assumption in the following implementation: the preprocessing pipeline does not
        modify the number of rows.
        """

        if index > len(self) - 1:
            raise IndexError("single positional indexer is out-of-bounds")

        # Compute actual index
        # -------------------
        augmented_df_id = np.searchsorted(
            getattr(self, "augmented_df_lengths_cumsum"), index, side="right"
        )
        augmented_df_index = (
            index - getattr(self, "augmented_df_lengths_cumsum")[augmented_df_id - 1]
            if augmented_df_id > 0
            else index
        )
        actual_index = getattr(self, "augmented_df_indices")[augmented_df_id][augmented_df_index]
        row_number = getattr(self, "indices_rows_mapping")[actual_index]

        # Get raw features
        # -------------------
        raw_features: DataFrameRow = self.data[row_number]

        # Apply scaling factor
        # -------------------
        scaling_factor = self.data_augmentation_scaling_factors[augmented_df_id]
        raw_features = raw_features.set(
            column=self.augmented_col,
            value=scaling_factor * raw_features[self.augmented_col],
        )

        # Apply transformations
        # -------------------
        sample = (
            DataFrameRow.from_single_row_df(
                self.fitted_preprocessing_pipeline.transform(raw_features.single_row_df)
            )
            if self.fitted_preprocessing_pipeline
            else raw_features
        )

        # Add label
        # -------------------
        if self.is_success is not None:
            sample = sample.set(
                column=self.label_col, value=self.is_success, dtype=self.labels_dtype
            )

        # Add weights back if removed by preprocessing pipeline
        # -------------------
        if (self.sample_weight_col is not None) and (self.sample_weight_col not in sample):
            sample = sample.set(
                column=self.sample_weight_col,
                value=raw_features[self.sample_weight_col],
                dtype=raw_features.dtypes[self.sample_weight_col],
            )

        return sample

    def __len__(self):
        return self.augmented_df_lengths_cumsum[-1]
