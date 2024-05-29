from __future__ import annotations

import copy
from typing import Any, Dict, List, Literal, Optional, Set, Union, cast, no_type_check

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.saving import _load_from_checkpoint
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from pytorch_utils.logging.loggers import (
    Logger,
    VoidLogger,
    is_logging_disabled,
    use_loggers,
)
from pytorch_utils.pandas.utils import DataFrameRow, ListDataFrameRows
from pytorch_utils.dataset_configurations import (
    AugmentedBernoulliDatasetConfigs,
    DataAugmentationConfig,
)
from pytorch_utils.datasets import (
    AugmentedBernoulliDataset,
    MLStage,
)
from pytorch_utils.utils import (
    BatchTorchTensors,
    NamedTorchTensors,
)


class AugmentedBernoulliDataModule(pl.LightningDataModule):
    """
    For prediction: attributes `prediction_df` (and
    optionally `prediction_scaling_factors`,
    `prediction_min_augmented_value` and
    `prediction_max_augmented_value`) must be set as desired.
    """

    def __init__(
        self,
        augmented_bernoulli_dataset_configs: AugmentedBernoulliDatasetConfigs,
        preprocessing_pipeline: Optional[Pipeline] = None,
        train_dataloader_params: Dict[str, Any] = {},
        val_dataloader_params: Dict[str, Any] = {},
        test_dataloader_params: Dict[str, Any] = {},
        predict_dataloader_params: Dict[str, Any] = {},
        data_module_logger: Logger = VoidLogger(),
        preprocessing_pipeline_logger: Logger = VoidLogger(),
        prepare_data_per_node: bool = False,
    ):
        super().__init__()

        # Set attributes
        # -------------------
        self.prepare_data_per_node = prepare_data_per_node

        self.augmented_bernoulli_dataset_configs = augmented_bernoulli_dataset_configs

        self.augmented_col = self.augmented_bernoulli_dataset_configs.augmented_col
        self.label_col = self.augmented_bernoulli_dataset_configs.label_col
        self.sample_weight_col = self.augmented_bernoulli_dataset_configs.sample_weight_col

        self.preprocessing_pipeline = preprocessing_pipeline

        # Copying the dataloader params before updating them is safer (it avoids
        # modifying the same underlying object when the same dictionary is passed
        # as input several times: see below with collate_fn and shuffle)
        self.train_dataloader_params = copy.deepcopy(train_dataloader_params)
        self.val_dataloader_params = copy.deepcopy(val_dataloader_params)
        self.test_dataloader_params = copy.deepcopy(test_dataloader_params)
        self.predict_dataloader_params = copy.deepcopy(predict_dataloader_params)

        self._batch_size: Optional[int] = None
        # see property batch_size and associated setter below

        self.loggers = {
            "data_module_logger": data_module_logger,
            "preprocessing_pipeline_logger": preprocessing_pipeline_logger,
        }

        # Teh following attributes must be set depending on what we wish to predict
        self.prediction_df: Optional[pd.DataFrame] = None
        self.prediction_scaling_factors: np.ndarray = np.array([1.0])
        self.prediction_min_augmented_value: float = -float("inf")
        self.prediction_max_augmented_value: float = float("inf")

        self._input_features_dtypes: Optional[pd.Series] = (
            None  # features dtypes before preprocessing
        )
        self._output_features_dtypes: Optional[pd.Series] = (
            None  # features dtypes after preprocessing
        )
        self._output_real_features: Optional[List[str]] = None
        self._output_categorical_features: Optional[Dict[str, Any]] = None

        self.prediction_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.validation_set: Optional[Dataset] = None
        self.train_set: Optional[Dataset] = None

        # Set pipeline outputs to pandas dataframes
        # -------------------
        if self.preprocessing_pipeline:
            self.preprocessing_pipeline.set_output(transform="pandas")

        # Overwrite collate_fn with pandas-specific version
        # -------------------
        self.train_dataloader_params.update({"collate_fn": self.preprocessing_pandas_collate_fn})
        self.val_dataloader_params.update({"collate_fn": self.preprocessing_pandas_collate_fn})
        self.test_dataloader_params.update({"collate_fn": self.preprocessing_pandas_collate_fn})
        self.predict_dataloader_params.update(
            {
                "shuffle": False,  # never shuffle when predicting
                "collate_fn": self.preprocessing_pandas_collate_fn,
            }
        )

        # Save hyperparameters
        # -------------------
        self.save_hyperparameters()  # saves all constructor params by default

    def preprocessing_pandas_collate_fn(
        self, batch: List[DataFrameRow]
    ) -> Union[NamedTorchTensors, BatchTorchTensors]:
        """
        More computationally efficient than using `collate_fn=lambda batch: pd.concat(batch, axis=0, sort=True)` and
        setting `fitted_transformers_pipeline=self.preprocessing_pipeline` in AugmentedBernoulliDataset. This
        vectorized the transform operations.
        """
        # collated_batch = pd.concat(batch, axis=0, sort=True)
        collated_batch = ListDataFrameRows.from_list_rows(batch).dataframe
        return self.transform_to_tensors(collated_batch)

    def transform(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Implicit assumption in the following implementation: the preprocessing pipeline does not
        modify the number of rows.
        """
        contains_labels = self.label_col in batch
        contains_weights = (self.sample_weight_col is not None) and (
            self.sample_weight_col in batch
        )
        non_feature_cols = (
            []
            + ([self.label_col] if contains_labels else [])
            + ([self.sample_weight_col] if contains_weights else [])
        )
        features = batch.drop(columns=non_feature_cols)
        transformed_batch = (
            self.preprocessing_pipeline.transform(features)
            if self.preprocessing_pipeline
            else features
        )
        if contains_labels:
            transformed_batch[self.label_col] = batch[self.label_col]
        if contains_weights and (self.sample_weight_col not in transformed_batch):
            # add back sample_weight_col if removed by preprocessing pipeline
            transformed_batch[self.sample_weight_col] = batch[self.sample_weight_col]
        return transformed_batch

    def transform_to_tensors(
        self, batch: pd.DataFrame
    ) -> Union[NamedTorchTensors, BatchTorchTensors]:
        return self.format_to_tensors(self.transform(batch))

    def augment_transform_to_tensors(
        self,
        batch: pd.DataFrame,
        augmentation_scaling_factors: np.ndarray = np.array([1.0]),
        min_augmented_value: float = -float("inf"),
        max_augmented_value: float = float("inf"),
    ) -> Union[NamedTorchTensors, BatchTorchTensors]:
        bb_conv_data_augmentation_conf = DataAugmentationConfig(
            augmented_col=self.augmented_col,
            scaling_factors=augmentation_scaling_factors,
            min_value=min_augmented_value,
            max_value=max_augmented_value,
        )
        return self.transform_to_tensors(bb_conv_data_augmentation_conf.augment_data(batch))

    def format_to_tensors(
        self, transformed_batch: pd.DataFrame
    ) -> Union[NamedTorchTensors, BatchTorchTensors]:
        contains_labels = self.label_col in transformed_batch
        contains_weights = (self.sample_weight_col is not None) and (
            self.sample_weight_col in transformed_batch
        )
        non_feature_cols = (
            []
            + ([self.label_col] if contains_labels else [])
            + ([self.sample_weight_col] if contains_weights else [])
        )
        formatted_features = self._format_pandas_dataframe(
            transformed_batch.drop(columns=non_feature_cols)
        )
        if contains_labels:
            formatted_labels = self._format_pandas_series(transformed_batch[self.label_col])
            formatted_weights = (
                self._format_pandas_series(transformed_batch[self.sample_weight_col])
                if contains_weights
                else None
            )
            return (formatted_features, formatted_labels, formatted_weights)
        else:
            return formatted_features

    def _format_pandas_dataframe(self, dataframe: pd.DataFrame) -> NamedTorchTensors:
        return {
            f: torch.from_numpy(
                dataframe[
                    # 1D tensors for categorical features, 2D for all other features
                    f if f in self.output_categorical_features else [f]
                ].to_numpy()
            )
            for f in dataframe
        }

    def _format_pandas_series(self, series: pd.Series) -> torch.Tensor:
        return torch.from_numpy(series.to_numpy())

    @property
    def is_preprocessing_pipeline_fitted(self) -> bool:
        """
        Boolean indicating whether pipeline is fitted
        """
        if self.preprocessing_pipeline:
            try:
                check_is_fitted(self.preprocessing_pipeline)
                is_preprocessing_pipeline_fitted = True
            except NotFittedError:
                is_preprocessing_pipeline_fitted = False
            return is_preprocessing_pipeline_fitted
        else:
            return True

    def fit_preprocessing_pipeline(self, refit: bool = False) -> None:
        if (refit or not self.is_preprocessing_pipeline_fitted) and self.preprocessing_pipeline:
            self._fit_preprocessing_pipeline()

    @use_loggers("preprocessing_pipeline_logger")
    def _fit_preprocessing_pipeline(self, logger: Logger) -> None:
        """
        Fit training pipeline if not None (otherwise do nothing).
        """
        logger.sklearn_autolog(disable=is_logging_disabled())
        if self.preprocessing_pipeline is not None:
            self.preprocessing_pipeline.fit(
                self.augmented_bernoulli_dataset_configs.all_training_data.drop(
                    columns=[self.label_col]
                    + ([self.sample_weight_col] if self.sample_weight_col is not None else [])
                )
            )
        logger.sklearn_autolog(disable=True)

    @property
    def input_features_dtypes(self) -> pd.Series:
        if self._input_features_dtypes is None:
            self._input_features_dtypes = (
                self.augmented_bernoulli_dataset_configs.sample(10, replace=True)
                .drop(
                    columns=[self.label_col]
                    + ([self.sample_weight_col] if self.sample_weight_col is not None else [])
                )
                .dtypes
            )

        return self._input_features_dtypes

    @property
    def output_features_dtypes(self) -> pd.Series:
        if self._output_features_dtypes is None:
            self.fit_preprocessing_pipeline(refit=False)

            self._output_features_dtypes = self.transform(
                self.augmented_bernoulli_dataset_configs.sample(10, replace=True).drop(
                    columns=[self.label_col]
                    + ([self.sample_weight_col] if self.sample_weight_col is not None else [])
                )
            ).dtypes

        return self._output_features_dtypes

    @property
    def output_features(self):
        return sorted(self.output_features_dtypes.index.to_list())

    @property
    def output_real_features(self) -> List[str]:
        if self._output_real_features is None:
            self._output_real_features = sorted(
                self.output_features_dtypes[
                    self.output_features_dtypes.map(lambda x: np.issubdtype(x, np.floating))
                ].index
            )

        return self._output_real_features

    @property
    def output_categorical_features(self) -> Dict[str, Set[int]]:
        if self._output_categorical_features is None:
            categorical_features_names = sorted(
                self.output_features_dtypes[
                    self.output_features_dtypes.map(lambda x: np.issubdtype(x, np.integer))
                ].index
            )

            training_data = self.augmented_bernoulli_dataset_configs.all_training_data[
                categorical_features_names
            ]

            self._output_categorical_features = {
                f: training_data[f].unique() for f in categorical_features_names
            }

        return self._output_categorical_features

    def prepare_data(self) -> None:
        # Fit preprocessing pipeline
        # -------------------
        # Preferable to fit the pipeline in `prepare_data` rather than in `setup` since in case of
        # parallel data loading, the `setup` method will be called once in every thread which is useless
        self.fit_preprocessing_pipeline(refit=True)
        self.log()

    def setup(self, stage: str) -> None:
        try:
            MLStage[stage]
        except KeyError:
            raise NotImplementedError(
                f"Argument stage should be either 'fit', 'validate', 'test' or 'predict' but '{stage}' was given."
            )

        if stage == MLStage.predict.name and self.prediction_df is None:
            raise ValueError("Attribute prediction_df must be set before predict setup.")

        # Fit preprocessing pipeline
        self.fit_preprocessing_pipeline(refit=False)

        # Create dataset corresponding to ml stage
        dataset = (
            AugmentedBernoulliDataset(
                data=ListDataFrameRows.from_pandas_df(self.prediction_df),
                is_success=None,  # labels are unknown in this case
                augmented_col=self.augmented_col,
                fitted_preprocessing_pipeline=None,
                data_augmentation_scaling_factors=self.prediction_scaling_factors,
                label_col=self.label_col,
                labels_dtype=np.dtype("int32"),
                sample_weight_col=None,
                min_augmented_value=self.prediction_min_augmented_value,
                max_augmented_value=self.prediction_max_augmented_value,
            )
            if stage == MLStage.predict.name
            else (
                ConcatDataset(
                    [
                        AugmentedBernoulliDataset.from_config(
                            config=ds_config,
                            ml_stage=MLStage[stage],  # type: ignore
                            fitted_preprocessing_pipeline=None,
                            label_col=self.label_col,
                            labels_dtype=np.dtype("int32"),
                            sample_weight_col=None,
                        )
                        for ds_config in self.augmented_bernoulli_dataset_configs
                    ]
                )
            )
        )

        # For the fit stage, also create validation set
        validation_dataset: Optional[Dataset] = (
            ConcatDataset(
                [
                    AugmentedBernoulliDataset.from_config(
                        config=ds_config,
                        ml_stage=MLStage.validate,
                        fitted_preprocessing_pipeline=None,
                        label_col=self.label_col,
                        labels_dtype=np.dtype("int32"),
                        sample_weight_col=None,
                    )
                    for ds_config in self.augmented_bernoulli_dataset_configs
                ]
            )
            if stage == MLStage.fit.name
            else None
        )

        # Set dataset attribute
        self.setup_datasets(MLStage[stage], dataset, validation_dataset)

    def setup_datasets(
        self,
        ml_stage: Literal[MLStage.fit, MLStage.validate, MLStage.test, MLStage.predict],
        dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
    ) -> None:
        if ml_stage is MLStage.fit:
            self.training_set = dataset
            if validation_dataset is None:
                raise ValueError("validation_dataset should not be None when ml_stage=MLStage.fit")
            self.validation_set = validation_dataset
        elif ml_stage is MLStage.validate:
            self.validation_set = dataset
        elif ml_stage is MLStage.test:
            self.test_set = dataset
        elif ml_stage is MLStage.predict:
            self.prediction_set = dataset
        else:
            raise NotImplementedError(f"Stage {ml_stage} is not implemented")

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        """
        Updates batch size (useful to apply `pytorch_lightning.trainer.Trainer.tune.scale_batch_size`)
        """
        self._batch_size = batch_size

        # update batch size of all dataloaders
        self.train_dataloader_params.update({"batch_size": batch_size})
        self.val_dataloader_params.update({"batch_size": batch_size})
        self.test_dataloader_params.update({"batch_size": batch_size})
        self.predict_dataloader_params.update({"batch_size": batch_size})

    def train_dataloader(self):
        return DataLoader(self.training_set, **self.train_dataloader_params)

    def val_dataloader(self):
        return DataLoader(self.validation_set, **self.val_dataloader_params)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self.test_dataloader_params)

    def predict_dataloader(self):
        return DataLoader(self.prediction_set, **self.predict_dataloader_params)

    def clear_data(self):
        self.input_features_dtypes  # just to make sure features have been saved before erasing data
        self.output_categorical_features  # just to make sure categories have been computed before erasing data
        self.augmented_bernoulli_dataset_configs.clear_data()
        if hasattr(self, "training_set"):
            del self.training_set
        if hasattr(self, "validation_set"):
            del self.validation_set
        if hasattr(self, "test_set"):
            del self.test_set
        if hasattr(self, "prediction_set"):
            del self.prediction_set
        return self

    @use_loggers("data_module_logger")
    def log(self, logger: Logger) -> None:
        self.augmented_bernoulli_dataset_configs.log()

        logger.log_param("augmented_col", self.augmented_col)
        logger.log_param("label_col", self.label_col)
        logger.log_param("sample_weight_col", self.sample_weight_col)
        logger.log_param("prepare_data_per_node", self.prepare_data_per_node)
        logger.log_param("batch_size", self.batch_size)
        logger.log_params(
            {f"train_dataloader__{k}": v for k, v in self.train_dataloader_params.items()}
        )
        logger.log_params(
            {f"val_dataloader__{k}": v for k, v in self.val_dataloader_params.items()}
        )
        logger.log_params(
            {f"test_dataloader__{k}": v for k, v in self.test_dataloader_params.items()}
        )
        logger.log_params(
            {f"predict_dataloader__{k}": v for k, v in self.predict_dataloader_params.items()}
        )

        if self.is_preprocessing_pipeline_fitted:
            logger.log_param("real_features", self.output_real_features)
            logger.log_pandas_artifact(
                self.input_features_dtypes.to_frame(), "input_features_dtypes"
            )
            logger.log_pandas_artifact(
                self.output_features_dtypes.to_frame(), "output_features_dtypes"
            )
            categorical_features_cardinals = pd.Series(
                {k: len(v) for k, v in self.output_categorical_features.items()},
                name="cardinals",
            )
            logger.log_pandas_artifact(
                categorical_features_cardinals.to_frame(),
                "categorical_features_cardinals",
            )

    @no_type_check
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        **kwargs: Any,
    ) -> AugmentedBernoulliDataModule:
        """
        We override this method to correct a bug with `map_location` argument.
        See Github issue: https://github.com/Lightning-AI/lightning/issues/17945
        """
        loaded = _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
        )
        return cast(AugmentedBernoulliDataModule, loaded)
