import os

import pandas as pd

import pytest

from flasc.flasc_dataframe import FlascDataFrame

test_data_dict = {
    "time":[0, 10, 20],
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": [7, 8, 9]
}

test_name_map = {"a": "AA"}


def test_type():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    assert isinstance(df, FlascDataFrame)

    # Assert df is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)

def test_check_flasc_format():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)

    # Should not raise an error
    df.check_flasc_format()

    # Convert to non-flasc format; should now raise an error
    df._user_format = "long"
    df.convert_to_user_format(inplace=True)
    with pytest.raises(ValueError):
        df.check_flasc_format()

def test_convert_to_long_format():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    df._user_format = "long" # Should be detected internally
    df.convert_to_user_format(inplace=True) # Should not pass

    # Check operation not allowed if no "time" column
    df.convert_to_flasc_format(inplace=True)
    df.drop(columns="time", inplace=True)
    with pytest.raises(ValueError):
        df.convert_to_user_format(inplace=True)

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

def test_n_turbines():
    # Currently, n_turbines based only on number of pow columns
    name_map = {"a": "pow_000", "b": "pow_001", "c": "ws_000"}
    df = FlascDataFrame(test_data_dict, name_map=name_map)
    assert df.n_turbines == 2

    name_map = {"a": "pow_000", "b": "ws_000", "c": "ws_001"}
    df = FlascDataFrame(test_data_dict, name_map=name_map)
    assert df.n_turbines == 1

    # Check n_turbines not valid if not in flasc format
    df.convert_to_user_format(inplace=True)
    with pytest.raises(ValueError):
        df.n_turbines
