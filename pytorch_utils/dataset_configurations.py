from __future__ import annotations

import copy
import itertools
from collections import UserList
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

# TYPE_CHECKING is False at runtime, but treated as True by type checkers.
# So the Python interpreters won't do pyspark import, but the type checker
# will still understand where it came from!
if TYPE_CHECKING:
    import pyspark.sql

    # We do not want to import pyspark at runtime, only when type checking (when building this project).
    # Because we want to get rid as much as possible of unnecessary (heavy) dependencies
    # in order to avoid having to install them too when installing this ML project (ex: for the inference part -> API).

from pytorch_utils.exceptions import InconsistentDatasetConfigurations
from pytorch_utils.logging.loggers import (
    Logger,
    VoidLogger,
    SingleLoggerDataclassLoggable,
    use_loggers,
)
from pytorch_utils.pandas.data_formatting import (
    PandasFormatter,
    PandasIdentityFormatter,
)
# from pytorch_utils.io.interface import MetaDataFrame


@dataclass(frozen=True)
class DataSplitConfig(SingleLoggerDataclassLoggable):
    """
    Configuration used to specify train, validation and test splits.
    The proportions should all lie in [0,1], with their sum smaller or equal to 1.
    In case the sum is strictly smaller than 1, only a random subset of the data is used.

    Attributes
    ----------
    training_proportion : float, default=1.
        proportion of dataframe to be used as training samples
    validation_proportion : float, default=0.
        proportion of dataframe to be used as validation samples
    test_proportion : float, default=0.
        proportion of dataframe to be used as test samples
    random_seed : Optional[int], default=None
        random seed used for random splitting
    stratify : Optional[List[str]], default=None
        list of column names used to stratify the data (see also sklearn.model_selection.train_test_split)
    """

    training_proportion: float = 1.0
    validation_proportion: float = 0.0
    test_proportion: float = 0.0
    random_seed: Optional[int] = None
    stratify: Optional[List[str]] = None
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def __post_init__(self):
        # Check attribute values
        if not 0.0 <= self.training_proportion <= 1.0:
            raise ValueError("training_proportion should belong to unit interval [0,1]")
        if not 0.0 <= self.validation_proportion <= 1.0:
            raise ValueError("validation_proportion should belong to unit interval [0,1]")
        if not 0.0 <= self.test_proportion <= 1.0:
            raise ValueError("test_proportion should belong to unit interval [0,1]")
        if (
            not self.training_proportion + self.validation_proportion + self.test_proportion
            <= 1.0 + 1e-6
        ):
            raise ValueError("The sum of all proportions should be at most 1")

        # Sort columns used to stratify splits
        # This ensures that two dataclasses that only differ by the ordering of `stratify` are considered equal
        object.__setattr__(
            self,
            "stratify",
            sorted(self.stratify) if self.stratify is not None else self.stratify,
        )

        # Make this class comply with Loggable Protocol
        object.__setattr__(self, "loggers", {"logger": self.logger})

    def train_valid_test_split(self, df: pd.DataFrame) -> Tuple[pd.Index, pd.Index, pd.Index]:
        def grouped_df(df: pd.DataFrame) -> pd.DataFrame:
            return (
                df.assign(
                    **{
                        # In case some values in self.stratify are null,
                        # we need to specify dropna=False to pandas groupby method
                        # but this causes some aggregation function (like "sample") to raise
                        # a KeyError. So the alternative method we found is to first
                        # replace null values by values with the same dtype that are not
                        # already present. N.B: this will work for both strings and numeric types.
                        f"{col}_bis_temp": df[col].fillna(df[col].max() * 2)
                        for col in self.stratify
                    }
                ).groupby(by=[f"{col}_bis_temp" for col in self.stratify])
                if self.stratify
                else df
            )

        training_indices = (
            grouped_df(df)
            .sample(frac=self.training_proportion, random_state=self.random_seed)
            .index
        )

        val_test_data = df[~df.index.isin(training_indices)]

        validation_indices = (
            grouped_df(val_test_data)
            .sample(
                frac=min(
                    self.validation_proportion / (1 - self.training_proportion)
                    if self.validation_proportion > 0
                    else 0.0,
                    1.0,
                ),
                random_state=self.random_seed,
            )
            .index
        )

        test_data = df[~df.index.isin(training_indices.union(validation_indices))]

        test_indices = (
            grouped_df(test_data)
            .sample(
                frac=min(
                    self.test_proportion
                    / (1 - self.training_proportion - self.validation_proportion)
                    if self.test_proportion > 0
                    else 0.0,
                    1.0,
                ),
                random_state=self.random_seed,
            )
            .index
        )

        return training_indices, validation_indices, test_indices


@dataclass(frozen=True)
class DataAugmentationConfig(SingleLoggerDataclassLoggable):
    """
    Configuration used to specify data augmentation on specific column (`augmented_col`).
    The idea of this data augmentation is to duplicate the data several times, with only `augmented_col` changed
    by multiplying the original values with a scaling factor.

    This can be useful when there is some monotone relationship between a covariate (`augmented_col`) and the
    success of an event (0=success, 1=event).

    Attributes
    ----------
    augmented_col : str
        Name of column to augment
    scaling_factors : np.ndarray
        Numpy array of floats corresponding to scaling factors used for data augmentation
    """

    augmented_col: str
    scaling_factors: np.ndarray = np.array([1.0])
    min_value: float = -float("inf")
    max_value: float = float("inf")
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def __post_init__(self):
        # Sort scaling factors
        # This ensures that two dataclasses that only differ by the ordering of `scaling_factors` are considered equal
        object.__setattr__(self, "scaling_factors", np.sort(self.scaling_factors))

        # Make this class comply with Loggable Protocol
        object.__setattr__(self, "loggers", {"logger": self.logger})

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                self.scale_col(
                    data,
                    scaling_factor,
                    self.augmented_col,
                    min_value=self.min_value,
                    max_value=self.max_value,
                )
                for scaling_factor in self.scaling_factors
            ],
            axis=0,
            ignore_index=True,
            sort=True,
        )

    @staticmethod
    def scaling_filter(
        data: pd.DataFrame,
        scaling_factor: np.ndarray,
        col: str,
        min_value: float = -float("inf"),
        max_value: float = float("inf"),
    ) -> pd.DataFrame:
        """
        Filter used to drop scaled values that are outside the range [min_value, max_value].
        See method `scale_col`.
        """
        return (data[col] * scaling_factor >= min_value) & (data[col] * scaling_factor <= max_value)

    @staticmethod
    def scale_col(
        data: pd.DataFrame,
        scaling_factor: np.ndarray,
        col: str,
        min_value: float = -float("inf"),
        max_value: float = float("inf"),
    ) -> pd.DataFrame:
        """
        Return pandas dataframe identical to `data` with column `col` scaled by `scaling_factor`.
        The scaled values that are outside the range [min_value, max_value] are dropped.
        """
        scale_filter = DataAugmentationConfig.scaling_filter(
            data, scaling_factor, col, min_value, max_value
        )
        output_df = data[scale_filter].copy()
        output_df[col] *= scaling_factor
        return output_df

    @staticmethod
    def scaling_length(
        data: pd.DataFrame,
        scaling_factor: np.ndarray,
        col: str,
        min_value: float = -float("inf"),
        max_value: float = float("inf"),
    ) -> int:
        """
        Length of the pandas dataframe obtained when calling method `scale_col` with the exact same input.
        The implementation does not require explicitly building the dataframe.
        """
        scale_filter = DataAugmentationConfig.scaling_filter(
            data, scaling_factor, col, min_value, max_value
        )
        return scale_filter.sum()

    def augmentation_length(self, data: pd.DataFrame) -> int:
        """
        Length of the pandas dataframe obtained when calling method `augment_data` with the exact same input.
        The implementation does not require explicitly building the dataframe.
        """
        return sum(
            [
                self.scaling_length(
                    data,
                    scaling_factor,
                    self.augmented_col,
                    min_value=self.min_value,
                    max_value=self.max_value,
                )
                for scaling_factor in self.scaling_factors
            ]
        )

    @use_loggers("logger")
    def log(self, logger: Logger) -> None:
        params = asdict(self)
        del params["logger"]
        logger.log_params(params)


@dataclass(frozen=True)
class AugmentedBernoulliDatasetConfig(SingleLoggerDataclassLoggable):
    """
    Dataset configuration for augmented Bernoulli samples (binary outcomes: successful or not).
    There are 2 ways to construct an instance of `AugmentedBernoulliDatasetConfig`:
        1) either by calling the constructor and passing a pandas dataframe (with optional meatada) as input
        2) or by calling `class method `from_meta_dataframe` and passing a delta table as input
    Method 1 is preferred for testing/debugging/prototyping while method 2 is preferred for
    production and traceable experimentations (clean metadata, etc...).

    Attributes
    ----------
        data : pandas.DataFrame
            The Pandas dataframe containing the data.
        is_success : bool
            Whether the samples correspond to successful events or not
        split_config : DataSplitConfig
            The configuration for splitting between train, validation and test
        data_augmentation_config : DataAugmentationConfig
            The configuration for data augmentation
        metadata : Dict[str, Any], default={}
            Any information regarding the source data that we wish to track/save
    """

    data: pd.DataFrame = field(repr=False)
    is_success: bool
    data_augmentation_config: DataAugmentationConfig
    split_config: DataSplitConfig = DataSplitConfig()
    metadata: Dict[str, Any] = field(default_factory=dict)
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def __post_init__(self):
        # Compute training, validation and test indices
        (
            training_indices,
            validation_indices,
            test_indices,
        ) = self.split_config.train_valid_test_split(self.data)
        # N.B: sorting pandas data according to all columns (sorted in alphabetic order) for reproducibility
        object.__setattr__(
            self, "data", self.data.sort_values(by=self.columns).reset_index(drop=True)
        )

        object.__setattr__(self, "training_indices", training_indices)
        object.__setattr__(self, "validation_indices", validation_indices)
        object.__setattr__(self, "test_indices", test_indices)

        # Make this class comply with Loggable Protocol
        object.__setattr__(self, "loggers", {"logger": self.logger})

    @classmethod
    def from_meta_dataframe(
        cls,
        meta_df: Any[pyspark.sql.DataFrame],
        is_success: bool,
        data_augmentation_config: DataAugmentationConfig,
        split_config: DataSplitConfig,
        logger: Logger = VoidLogger(),
        spark_filter: Optional[str] = None,
        pandas_formatter: PandasFormatter = PandasIdentityFormatter(),
    ) -> AugmentedBernoulliDatasetConfig:
        """
        Use this method to construct an instance of `AugmentedBernoulliDatasetConfig` directly from a delta table
        with proper metadata.

        Attributes
        ----------
        delta_table : delta.tables.DeltaTable
            The delta table containing the data.
            To use a previous version of the data, call `restoreToVersion(version: int)`
            on `delta_table` before passing it to `AugmentedBernoulliDatasetConfig`.
        is_success : bool
            Whether the samples correspond to successful events or not
        split_config : DataSplitConfig
            The configuration for splitting between train, validation and test
        data_augmentation_config : DataAugmentationConfig
            The configuration for data augmentation
        pandas_formatter: PandasFormatter
            Any formatting on pandas data (cast dtypes, etc...).
        """
        metadata = copy.deepcopy(meta_df.metadata)
        metadata["common_spark_filter"] = spark_filter
        metadata["pandas_formatter"] = str(pandas_formatter)
        spark_filter = spark_filter if spark_filter is not None else "True"

        dataset_config = cls(
            data=pandas_formatter.format(meta_df.read().filter(spark_filter).toPandas()),
            is_success=is_success,
            data_augmentation_config=data_augmentation_config,
            split_config=split_config,
            metadata=metadata,
            logger=logger,
        )
        # Set _pandas_formatter attribute to be able to log info later
        object.__setattr__(dataset_config, "_pandas_formatter", pandas_formatter)

        return dataset_config

    def sample(self, n: int, replace: bool = False):
        return self.data.sample(n, replace=replace)

    @property
    def training_data(self):
        return self.data.iloc[self.training_indices]

    @property
    def training_data_length(self):
        return len(self.training_indices)

    @property
    def validation_data(self):
        return self.data.iloc[self.validation_indices]

    @property
    def validation_data_length(self):
        return len(self.validation_indices)

    @property
    def test_data(self):
        return self.data.iloc[self.test_indices]

    @property
    def test_data_length(self):
        return len(self.test_indices)

    @property
    def augmented_data(self):
        return self.data_augmentation_config.augment_data(self.data)

    @property
    def augmented_data_length(self):
        return self.data_augmentation_config.augmentation_length(self.data)

    @property
    def augmented_training_data(self):
        return self.data_augmentation_config.augment_data(self.training_data)

    @property
    def augmented_training_data_length(self):
        return self.data_augmentation_config.augmentation_length(self.training_data)

    @property
    def augmented_validation_data(self):
        return self.data_augmentation_config.augment_data(self.validation_data)

    @property
    def augmented_validation_data_length(self):
        return self.data_augmentation_config.augmentation_length(self.validation_data)

    @property
    def augmented_test_data(self):
        return self.data_augmentation_config.augment_data(self.test_data)

    @property
    def augmented_test_data_length(self):
        return self.data_augmentation_config.augmentation_length(self.test_data)

    @property
    def augmented_col(self):
        return self.data_augmentation_config.augmented_col

    @property
    def data_augmentation_scaling_factors(self):
        return self.data_augmentation_config.scaling_factors

    @property
    def columns(self):
        return sorted(self.data.columns)

    @property
    def dtypes(self):
        return self.data.dtypes

    def __len__(self) -> int:
        return len(self.data)

    def clear_data(self) -> AugmentedBernoulliDatasetConfig:
        object.__setattr__(self, "data", self.data[:0])
        return self

    @use_loggers("logger")
    def log(self, logger: Logger) -> None:
        self.split_config.log()
        self.data_augmentation_config.log()
        logger.log_param("metadata", self.metadata)
        logger.log_param("is_success", self.is_success)
        logger.log_param("columns", self.columns)
        logger.log_metrics(
            {
                "total_length": len(self),
                "total_augmented_length": self.augmented_data_length,
                "training_data_length": self.training_data_length,
                "augmented_training_data_length": self.augmented_training_data_length,
                "validation_data_length": self.validation_data_length,
                "augmented_validation_data_length": self.augmented_validation_data_length,
                "test_data_length": self.test_data_length,
                "augmented_test_data_length": self.augmented_test_data_length,
            }
        )
        logger.log_pandas_artifact(self.dtypes.rename("data_dtypes").to_frame(), "data_dtypes")
        logger.log_pandas_artifact(self.data.describe(), "data_statistics")

        if hasattr(self, "_pandas_formatter"):
            _pandas_formatter: PandasFormatter = getattr(self, "_pandas_formatter")
            _pandas_formatter.log()


class AugmentedBernoulliDatasetConfigs(UserList):
    def __init__(
        self,
        augmented_bernoulli_dataset_configs=List[AugmentedBernoulliDatasetConfig],
        label_col: str = "success_labels",
        labels_dtype: Type[np.intc] = np.int32,
        sample_weight_col: Optional[str] = None,
        logger: Logger = VoidLogger(),
    ):
        self.data: List[AugmentedBernoulliDatasetConfig] = augmented_bernoulli_dataset_configs
        self.label_col = label_col
        self.labels_dtype = labels_dtype
        self.sample_weight_col = sample_weight_col
        self.check_compatibility()
        self.loggers = {"logger": logger}  # make this class comply with Loggable Protocol

    def check_compatibility(self) -> None:
        if len(self.data) == 0:
            raise InconsistentDatasetConfigurations(
                "`augmented_bernoulli_dataset_configs` should have at least one element"
            )

        elif len(self.data) > 1:
            if self.sample_weight_col and not all(
                [self.sample_weight_col in conf.columns for conf in self.data]
            ):
                raise InconsistentDatasetConfigurations(
                    "All instances of `AugmentedBernoulliDatasetConfig` should have column `sample_weight_col`."
                )

            if not all(
                [self.data[0].augmented_col == conf.augmented_col for conf in self.data[1:]]
            ):
                raise InconsistentDatasetConfigurations(
                    "All instances of `AugmentedBernoulliDatasetConfig` should have "
                    + "the same attribute `augmented_col`."
                )

            if not all([set(self.data[0].columns) == set(conf.columns) for conf in self.data[1:]]):
                raise InconsistentDatasetConfigurations(
                    "All instances of `AugmentedBernoulliDatasetConfig` should have the same `columns`."
                )

            if not all([(self.data[0].dtypes == conf.dtypes).all() for conf in self.data[1:]]):
                raise InconsistentDatasetConfigurations(
                    "All instances of `AugmentedBernoulliDatasetConfig` should have the same `dtypes`."
                )

    @property
    def augmented_col(self):
        return self.data[0].augmented_col

    @property
    def columns(self):
        return self.data[0].columns

    @property
    def dtypes(self):
        return self.data[0].dtypes

    def _concat_data(self, list_df: List[pd.DataFrame], list_labels: List[bool]) -> pd.DataFrame:
        output_df = pd.concat(
            [df for df in list_df if len(df) > 0],
            axis=0,
            sort=True,
        )
        output_df[self.label_col] = list(
            itertools.chain(*[[label] * len(df) for df, label in zip(list_df, list_labels)])
        )
        return output_df.astype({self.label_col: self.labels_dtype})

    def sample(self, n: int, replace: bool = False):
        return self._concat_data(
            [conf_df.sample(n, replace=replace) for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        ).sample(n, replace=replace)

    @property
    def all_data(self):
        return self._concat_data(
            [conf_df.data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_data_length(self):
        return sum([len(conf_df) for conf_df in self.data])

    @property
    def all_training_data(self):
        return self._concat_data(
            [conf_df.training_data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_training_data_length(self):
        return sum([conf_df.training_data_length for conf_df in self.data])

    @property
    def all_validation_data(self):
        return self._concat_data(
            [conf_df.validation_data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_validation_data_length(self):
        return sum([conf_df.validation_data_length for conf_df in self.data])

    @property
    def all_test_data(self):
        return self._concat_data(
            [conf_df.test_data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_test_data_length(self):
        return sum([conf_df.test_data_length for conf_df in self.data])

    @property
    def all_augmented_data(self):
        return self._concat_data(
            [conf_df.augmented_data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_augmented_data_length(self):
        return sum([conf_df.augmented_data_length for conf_df in self.data])

    @property
    def all_augmented_training_data(self):
        return self._concat_data(
            [conf_df.augmented_training_data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_augmented_training_data_length(self):
        return sum([conf_df.augmented_training_data_length for conf_df in self.data])

    @property
    def all_augmented_validation_data(self):
        return self._concat_data(
            [conf_df.augmented_validation_data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_augmented_validation_data_length(self):
        return sum([conf_df.augmented_validation_data_length for conf_df in self.data])

    @property
    def all_augmented_test_data(self):
        return self._concat_data(
            [conf_df.augmented_test_data for conf_df in self.data],
            [conf_df.is_success for conf_df in self.data],
        )

    @property
    def all_augmented_test_data_length(self):
        return sum([conf_df.augmented_test_data_length for conf_df in self.data])

    def clear_data(self) -> AugmentedBernoulliDatasetConfigs:
        self.data = [d.clear_data() for d in self.data]
        self.check_compatibility()
        return self

    @use_loggers("logger")
    def log(self, logger: Logger) -> None:
        for conf in self.data:
            conf.log()

        logger.log_param("label_col", self.label_col)
        logger.log_param("sample_weight_col", self.sample_weight_col)
        logger.log_param("augmented_col", self.augmented_col)
        logger.log_param("columns", self.columns)
        logger.log_metrics(
            {
                "all_data_length": self.all_data_length,
                "all_augmented_data_length": self.all_augmented_data_length,
                "all_training_data_length": self.all_training_data_length,
                "all_augmented_training_data_length": self.all_augmented_training_data_length,
                "all_validation_data_length": self.all_validation_data_length,
                "all_augmented_validation_data_length": self.all_augmented_validation_data_length,
                "all_test_data_length": self.all_test_data_length,
                "all_augmented_test_data_length": self.all_augmented_test_data_length,
            }
        )
        logger.log_pandas_artifact(self.dtypes.rename("data_dtypes").to_frame(), "data_dtypes")
