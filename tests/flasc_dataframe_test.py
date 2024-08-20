import pandas as pd

from flasc.flasc_dataframe import FlascDataFrame


def test_type():
    df = FlascDataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, name_map={"a": "AA"})
    assert isinstance(df, FlascDataFrame)

    # Assert df is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)


def test_passing_flasc_dataframe_to_flasc_dataframe():
    df = FlascDataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, name_map={"a": "AA"})
    df2 = FlascDataFrame(df)
    assert isinstance(df2, FlascDataFrame)

    # Assert df2 is a pandas DataFrame
    assert isinstance(df2, pd.DataFrame)

    # Assert df and df2 are equal
    assert df.equals(df2)
