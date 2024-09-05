import os

import pandas as pd
import pytest

from flasc.flasc_dataframe import FlascDataFrame

# Define dataframes in each format that relate through the test name map
test_wide_dict = {
    "time": [0, 10, 20],
    "pow_000": [0, 100, 200],
    "ws_000": [8, 8, 8],
    "pow_001": [50, 150, 250],
    "ws_001": [9, 9, 9],
}

test_name_map = {"T1PWR": "pow_000", "T1WS": "ws_000", "T2PWR": "pow_001", "T2WS": "ws_001"}


test_semi_wide_dict = {
    "time": [0, 0, 10, 10, 20, 20],
    "turbine_id": [0, 1, 0, 1, 0, 1],
    "pow": [0, 50, 100, 150, 200, 250],
    "ws": [8, 9, 8, 9, 8, 9],
}

test_long_dict = {
    "time": [0, 0, 0, 0, 10, 10, 10, 10, 20, 20, 20, 20],
    "variable": ["T1PWR", "T2PWR", "T1WS", "T2WS"] * 3,
    "value": [0, 50, 8, 9, 100, 150, 8, 9, 200, 250, 8, 9],
}

test_wide_user_dict = {
    "time": [0, 10, 20],
    "T1PWR": [0, 100, 200],
    "T1WS": [8, 8, 8],
    "T2PWR": [50, 150, 250],
    "T2WS": [9, 9, 9],
}


def test_type():
    df = FlascDataFrame(test_wide_dict, name_map=test_name_map)
    assert isinstance(df, FlascDataFrame)

    df2 = df.drop(columns="ws_001")  # Modifies the dataframe, returns a copy
    assert isinstance(df2, FlascDataFrame)

    # Assert df is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)


def test__metadata():
    df = FlascDataFrame(test_wide_dict, name_map=test_name_map)
    df._user_format = "long"
    df._in_flasc_format = False
    df2 = df.drop(columns="ws_001")  # Modifies the dataframe, returns a copy
    assert hasattr(df2, "name_map")
    assert df2.name_map == test_name_map
    assert hasattr(df2, "_user_format")
    assert df2._user_format == "long"
    assert hasattr(df2, "_in_flasc_format")
    assert df2._in_flasc_format == True  # Resets, since "_in_flasc_format" not in _metadata.
    # May want to add "_in_flasc_format" to _metadata in future, but this
    # demonstrates functionality


def test_printout():
    df = FlascDataFrame(test_wide_dict, name_map=test_name_map)
    df._in_flasc_format = True
    print(df)
    print("\n")
    df._in_flasc_format = False
    print(df)
    print("\n")
    print(df.head())  # In FLASC format, presumably because .head() returns a reinstantiated copy?


def test_check_flasc_format():
    df = FlascDataFrame(test_wide_dict, name_map=test_name_map)

    # Should not raise an error
    df.check_flasc_format()

    # Convert to non-flasc format; should now raise an error
    df._user_format = "long"
    df.convert_to_user_format(inplace=True)
    with pytest.raises(ValueError):
        df.check_flasc_format()


def test_convert_to_long_format():
    df_wide = FlascDataFrame(test_wide_dict, name_map=test_name_map)
    df_long_test = pd.DataFrame(test_long_dict)

    # Test conversion with return
    df_wide._user_format = "long"  # Should be detected internally
    df_wide_copy = df_wide.copy()
    df_long = df_wide.convert_to_user_format(inplace=False)

    # Test df_long is not in flasc format
    assert not df_long._in_flasc_format

    # Test returned frame is matched to expected value
    pd.testing.assert_frame_equal(df_long, df_long_test)

    # Test original frame is unchanged
    pd.testing.assert_frame_equal(df_wide, df_wide_copy)

    # Now test in place conversion
    df_wide.convert_to_user_format(inplace=True)
    pd.testing.assert_frame_equal(df_wide, df_long_test)

    # Assert not in flasc format
    assert not df_wide._in_flasc_format

    # Now test the back conversion
    df_back_to_wide = df_wide.convert_to_flasc_format(inplace=False)

    # Resort the columns to match
    df_back_to_wide = df_back_to_wide[df_wide_copy.columns]

    pd.testing.assert_frame_equal(df_back_to_wide, df_wide_copy)

    # Assert is in flasc format
    assert df_back_to_wide._in_flasc_format

    # Test in place version
    df_wide.convert_to_flasc_format(inplace=True)

    # Sort columns to match
    df_wide = df_wide[df_wide_copy.columns]

    pd.testing.assert_frame_equal(df_wide, df_wide_copy)

    # Check operation not allowed if no "time" column
    df_wide.drop(columns="time", inplace=True)
    with pytest.raises(ValueError):
        df_wide.convert_to_user_format(inplace=True)


def test_convert_to_wide_format():
    # Test wide to wide conversion

    pass


def test_pickle():
    df = FlascDataFrame(test_wide_dict)
    df.name_map = test_name_map
    df.to_pickle("test_pickle.pkl")

    df2 = pd.read_pickle("test_pickle.pkl")
    assert isinstance(df2, FlascDataFrame)
    assert df2.name_map == test_name_map

    os.remove("test_pickle.pkl")


def test_feather():
    df = FlascDataFrame(test_wide_dict, name_map=test_name_map)
    df.to_feather("test_feather.ftr")

    df2 = pd.read_feather("test_feather.ftr")
    # Loaded DataFrame is a pandas DataFrame, not a FlascDataFrame
    assert not isinstance(df2, FlascDataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert not hasattr(df2, "name_map")

    os.remove("test_feather.ftr")


def test_csv():
    df = FlascDataFrame(test_wide_dict, name_map=test_name_map)
    df.to_csv("test_csv.csv")

    df2 = pd.read_csv("test_csv.csv")
    # Loaded DataFrame is a pandas DataFrame, not a FlascDataFrame
    assert not isinstance(df2, FlascDataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert not hasattr(df2, "name_map")

    os.remove("test_csv.csv")


def test_n_turbines():
    # Currently, n_turbines based only on number of pow columns
    df = FlascDataFrame(test_wide_dict, name_map=test_name_map)
    assert df.n_turbines == 2

    # Check n_turbines not valid if not in flasc format
    df._user_format = "long"
    df.convert_to_user_format(inplace=True)
    with pytest.raises(ValueError):
        df.n_turbines
