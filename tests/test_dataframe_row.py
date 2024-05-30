from collections import OrderedDict
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, indexes
from hypothesis.strategies import integers

from pytorch_utils.pandas.utils import DataFrameRow, ListDataFrameRows

columns = [
    column("a", dtype=str),
    column("b", dtype=int),
    column("c", dtype=np.int8),
    column("d", dtype=float),
    column("e", dtype=np.float16),
]

dataframe = data_frames(
    columns=columns,
    index=indexes(
        elements=integers(min_value=0, max_value=500),
        dtype=int,
        min_size=20,
        max_size=21,
        unique=False,
    ),
)


@given(df=dataframe)
@settings(max_examples=100)
def test_df_row(df: pd.DataFrame):
    idx = 12

    # Test constructor
    with pytest.raises(ValueError) as _:
        DataFrameRow(OrderedDict(a=1), OrderedDict(a=np.dtype("int8"), b=np.dtype("float32")), 2)
    with pytest.raises(ValueError) as _:
        DataFrameRow.from_single_row_df(single_row_df=df)
    df_row = DataFrameRow.from_single_row_df(single_row_df=df.iloc[[idx]])

    # Test immutability
    with pytest.raises(FrozenInstanceError):
        df_row.values = 1  # type: ignore
    with pytest.raises(FrozenInstanceError):
        df_row.dtypes = 1  # type: ignore
    with pytest.raises(FrozenInstanceError):
        df_row.index = 1  # type: ignore

    # Check dtypes
    pd.testing.assert_series_equal(pd.Series(df_row.dtypes), df.dtypes)

    # Check columns
    assert all(df_row.columns == df.columns)

    assert all(
        [
            (df_row[col] == df.iloc[idx][col])
            or (not (isinstance((df.iloc[idx][col]), str)) and np.isnan(df.iloc[idx][col]))
            for col in list(df.columns)
        ]
    )

    # Check pandas conversion
    pd.testing.assert_frame_equal(df_row.single_row_df, df.iloc[[idx]])


@given(df=dataframe)
@settings(max_examples=100)
def test_df_row_list(df: pd.DataFrame):
    # Check constructor
    list_df_rows = ListDataFrameRows.from_pandas_df(df)

    # Check indices and columns
    assert all(list_df_rows.columns == df.columns)
    assert np.all(df.index.to_numpy() == np.array(list_df_rows.indices))

    # Check conversion to dataframe
    pd.testing.assert_frame_equal(
        df[["a", "c", "e"]], list_df_rows.build_dataframe(["a", "c", "e"])
    )

    pd.testing.assert_frame_equal(df, list_df_rows.dataframe)
