from __future__ import annotations

from collections import OrderedDict, UserList
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataFrameRow:
    """
    A single row of a pandas dataframe.
    The motivation for this abstraction is to speed up concatenation
    of a list of rows (see function `concat_df_rows`).
    Only integer indices are supported
    """

    values: OrderedDict[str, Any]
    dtypes: OrderedDict[str, np.dtype]
    index: int

    def __post_init__(self):
        if not self.values.keys() == self.dtypes.keys():
            raise ValueError("values and dtypes should have the same set of keys")

    @classmethod
    def from_single_row_df(cls, single_row_df: pd.DataFrame) -> DataFrameRow:
        if len(single_row_df) != 1:
            raise ValueError("single_row_df should contain exactly one row")
        return cls(
            values=single_row_df.iloc[0].to_dict(OrderedDict),
            dtypes=single_row_df.dtypes.to_dict(OrderedDict),
            index=single_row_df.index[0],
        )

    @property
    def columns(self) -> List[str]:
        return list(self.dtypes.keys())

    @property
    def single_row_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: np.array([self.values[column]], dtype=self.dtypes[column])
                for column in self.columns
            },
            index=[self.index],
        )

    def __getitem__(self, column: str) -> Any:
        return self.values[column]

    def __contains__(self, column: str) -> bool:
        return column in self.columns

    def set(self, column: str, value: Any, dtype: Optional[np.dtype] = None) -> DataFrameRow:
        new_values = OrderedDict(self.values)
        new_values.update({column: value})
        new_dtypes = OrderedDict(self.dtypes)
        if column not in self.columns or dtype:
            new_dtypes.update({column: np.array(value, dtype=dtype).dtype})
        return DataFrameRow(
            new_values,
            new_dtypes,
            self.index,
        )

    def __repr__(self) -> str:
        return (
            "{\n"
            + "\n".join(
                [
                    f"   {column} ({self.dtypes[column]}): {self.values[column]}"
                    for column in self.columns
                ]
            )
            + "\n}"
        )


class ListDataFrameRows(UserList[DataFrameRow]):
    """
    This class improves the speed of execution of some
    of the built-in operations of pandas dataframes like
    `__getitem__` (subscript),
    """

    def __init__(
        self,
        columns: List[str],
        dtypes: OrderedDict[str, np.dtype],
        data: List[DataFrameRow],
    ):
        """
        Avoid using the constructor directly, prefer `from_pandas_df`
        and `from_list_rows` methods.
        """
        self.columns = columns
        self.dtypes = dtypes
        self.data = data

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
    ):
        columns = df.columns
        dtypes = df.dtypes.to_dict(OrderedDict)
        column_values = {column: df[column].to_numpy() for column in df.columns}
        indices = df.index
        data: List[DataFrameRow] = [
            DataFrameRow(
                OrderedDict({column: column_values[column][i] for column in df.columns}),
                dtypes,
                indices[i],
            )
            for i in range(len(df))
        ]
        return cls(columns=columns, dtypes=dtypes, data=data)

    @classmethod
    def from_list_rows(cls, list_rows: List[DataFrameRow]):
        """
        Warning: not checking rows consistency to avoid slowing down execution.
        Check rows consistency before calling this method if needed.
        """
        return cls(columns=list_rows[0].columns, dtypes=list_rows[0].dtypes, data=list_rows)

    def get_column(self, column: str) -> np.ndarray:
        return np.array([row[column] for row in self], dtype=self.dtypes[column])

    def build_dataframe(self, columns: List[str]) -> pd.DataFrame:
        """
        Convert list of rows back to a pandas dataframe (only keeping
        columns in `columns`).
        Concatenating a list of `DataFrameRow` is much faster than
        concatenating a list of single-row `pandas.DataFrame`
        using `pandas.concat` function.
        """
        return pd.DataFrame(
            {column: self.get_column(column) for column in self.columns if column in columns},
            index=[row.index for row in self],
        )

    @property
    def indices(self):
        return [row.index for row in self]

    @property
    def dataframe(
        self,
    ) -> pd.DataFrame:
        return self.build_dataframe(self.columns)
