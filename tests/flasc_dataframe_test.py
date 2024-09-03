import os

import pandas as pd

from flasc.flasc_dataframe import FlascDataFrame

test_data_dict = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}

test_name_map = {"a": "AA"}


def test_type():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    assert isinstance(df, FlascDataFrame)

    # Assert df is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)


def test_pickle():
    df = FlascDataFrame(test_data_dict)
    df.name_map = test_name_map
    df.to_pickle("test_pickle.pkl")

    df2 = pd.read_pickle("test_pickle.pkl")
    assert isinstance(df2, FlascDataFrame)
    assert df2.name_map == test_name_map

    os.remove("test_pickle.pkl")


def test_feather():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    df.to_feather("test_feather.ftr")

    df2 = pd.read_feather("test_feather.ftr")
    # Loaded DataFrame is a pandas DataFrame, not a FlascDataFrame
    assert not isinstance(df2, FlascDataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert not hasattr(df2, "name_map")

    os.remove("test_feather.ftr")


def test_csv():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    df.to_csv("test_csv.csv")

    df2 = pd.read_csv("test_csv.csv")
    # Loaded DataFrame is a pandas DataFrame, not a FlascDataFrame
    assert not isinstance(df2, FlascDataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert not hasattr(df2, "name_map")

    os.remove("test_csv.csv")
