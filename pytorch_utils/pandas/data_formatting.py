from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, List, Protocol

import pandas as pd

from pytorch_utils.logging.loggers import (
    Logger,
    VoidLogger,
    SingleLoggerDataclassLoggable,
    use_loggers,
)


def cast_float_cols(df: pd.DataFrame, target_dtype: str) -> pd.DataFrame:
    """
    Cast all float columns to a target dtype.
    """
    # Identify columns with float dtype
    casted_df = df.copy()
    float_cols = [col for col in casted_df.columns if pd.api.types.is_float_dtype(casted_df[col])]

    # Cast to float32
    casted_df[float_cols] = casted_df[float_cols].astype(target_dtype)
    return casted_df


def cast_integer_cols(df: pd.DataFrame, target_dtype: str) -> pd.DataFrame:
    """
    Cast all integer columns to a target dtype.
    """
    # Identify columns with float dtype
    casted_df = df.copy()
    integer_cols = [
        col for col in casted_df.columns if pd.api.types.is_integer_dtype(casted_df[col])
    ]

    # Cast to float32
    casted_df[integer_cols] = casted_df[integer_cols].astype(target_dtype)
    return casted_df


class PandasFormatter(SingleLoggerDataclassLoggable, Protocol):
    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def __mul__(self, other_formatter: PandasFormatter) -> PandasFormatter:
        """
        Composition function (https://en.wikipedia.org/wiki/Function_composition)
        Allows to easily build "composite" formaters.
        `formater2 * formater1` returns a `PandasFormater` with a format method that
        first applies the `format` method of `formater1` and then applies the `format`
        method of `formater2` to the resulting pandas dataframe.
        Note that this operation is not commutative (like matrix multiplication).
        """
        return CompositeFormatter(formatters=[other_formatter, self], logger=self.logger).flatten()


@dataclass
class CompositeFormatter(PandasFormatter):
    formatters: List[PandasFormatter]
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies `format` method of all `formaters` sequentially starting from the first
        formater in the list.
        """
        return functools.reduce(lambda df, formatter: formatter.format(df), self.formatters, df)

    def _flatten_formatters_list(self) -> List[PandasFormatter]:
        flattened_formatters = []
        for formatter in self.formatters:
            if isinstance(formatter, CompositeFormatter):
                flattened_formatters += formatter._flatten_formatters_list()
            else:
                flattened_formatters += [formatter]
        return flattened_formatters

    def flatten(self) -> CompositeFormatter:
        return CompositeFormatter(formatters=self._flatten_formatters_list(), logger=self.logger)

    @use_loggers("logger")
    def log(self, logger: Logger, params_not_to_log: List[str] = ["logger"]) -> None:
        super().log.__wrapped__(self, logger, params_not_to_log=["logger", "formatters"])

        for formatter in self.formatters:
            formatter.log()


@dataclass
class PandasIdentityFormatter(PandasFormatter):
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


@dataclass
class PandasFloatCaster(PandasFormatter):
    target_dtype: str
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        return cast_float_cols(df, self.target_dtype)


@dataclass
class PandasIntegerCaster(PandasFormatter):
    target_dtype: str
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        return cast_integer_cols(df, self.target_dtype)


@dataclass
class PandasColumnsSelector(PandasFormatter):
    relevant_columns: List[str]
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.relevant_columns]


@dataclass
class PandasNullDropper(PandasFormatter):
    non_nullable_columns: List[str] = field(default_factory=list)
    nullable_columns: List[str] = field(default_factory=list)
    ignore_index: bool = True
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if self.non_nullable_columns and self.nullable_columns:
            raise ValueError("non_nullable_columns and nullable_columns are both non-empty")

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_subset = (
            self.non_nullable_columns
            if self.non_nullable_columns
            else (set(df.columns) - set(self.nullable_columns))
        )
        return df.dropna(subset=columns_subset, ignore_index=self.ignore_index)


@dataclass
class PandasValuesReplacer(PandasFormatter):
    """Applies `pandas.DataFrame.replace` with corresponding attributes"""

    to_replace: Any
    value: Any
    logger: Logger = field(default=VoidLogger(), repr=False, compare=False)

    def format(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.replace(to_replace=self.to_replace, value=self.value)
