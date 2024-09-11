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

test_channel_name_map = {"T1PWR": "pow_000", "T1WS": "ws_000", "T2PWR": "pow_001", "T2WS": "ws_001"}
test_channel_name_map_incomplete = {"T1PWR": "pow_000", "T1WS": "ws_000", "T2PWR": "pow_001"}
test_long_columns = {"variable_column": "variable", "value_column": "value"}


test_long_dict = {
    "time": [0, 0, 0, 0, 10, 10, 10, 10, 20, 20, 20, 20],
    "variable": ["T1PWR", "T2PWR", "T1WS", "T2WS"] * 3,
    "value": [0, 50, 8, 9, 100, 150, 8, 9, 200, 250, 8, 9],
}

test_long_dict_incomplete = {
    "time": [0, 0, 0, 0, 10, 10, 10, 10, 20, 20, 20, 20],
    "variable": ["T1PWR", "T2PWR", "T1WS", "ws_001"] * 3,
    "value": [0, 50, 8, 9, 100, 150, 8, 9, 200, 250, 8, 9],
}

test_wide_user_dict = {
    "time": [0, 10, 20],
    "T1PWR": [0, 100, 200],
    "T1WS": [8, 8, 8],
    "T2PWR": [50, 150, 250],
    "T2WS": [9, 9, 9],
}

test_wide_user_dict_incomplete = {
    "time": [0, 10, 20],
    "T1PWR": [0, 100, 200],
    "T1WS": [8, 8, 8],
    "T2PWR": [50, 150, 250],
    "ws_001": [9, 9, 9],
}


def assert_equal_except_row_col_order(df1, df2):
    # Sort the columns
    df_1_c = df1.sort_index(axis=1)
    df_2_c = df2.sort_index(axis=1)

    # If "variable" is a column, sort by ['time', 'variable']
    if "variable" in df_1_c.columns:
        df_1_c = df_1_c.sort_values(by=["time", "variable"]).reset_index(drop=True)
        df_2_c = df_2_c.sort_values(by=["time", "variable"]).reset_index(drop=True)

    else:
        df_1_c = df_1_c.sort_values(by=["time"]).reset_index(drop=True)
        df_2_c = df_2_c.sort_values(by=["time"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(df_1_c, df_2_c)


def test_type():
    df = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    assert isinstance(df, FlascDataFrame)

    df2 = df.drop(columns="ws_001")  # Modifies the dataframe, returns a copy
    assert isinstance(df2, FlascDataFrame)

    # Assert df is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)


def test__metadata():
    df = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )
    df2 = df.drop(columns="ws_001")  # Modifies the dataframe, returns a copy
    assert hasattr(df2, "channel_name_map")
    assert df2.channel_name_map == test_channel_name_map
    assert hasattr(df2, "_user_format")
    assert df2._user_format == "long"
    assert hasattr(df2, "in_flasc_format")
    assert df2.in_flasc_format == True


def test_printout():
    df = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    # df._in_flasc_format = True
    print(df)
    print("\n")
    # df._in_flasc_format = False
    # print(df)
    # print("\n")


def test_time_required():
    # Check that the time column is present
    with pytest.raises(ValueError):
        FlascDataFrame(
            {"pow_000": [0, 100, 200], "ws_000": [8, 8, 8]}, channel_name_map=test_channel_name_map
        )


def test_check_flasc_format():
    df = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)

    # Should not raise an error
    df.check_flasc_format()

    # Convert to non-flasc format; should now raise an error
    df.convert_to_user_format(inplace=True)
    with pytest.raises(ValueError):
        df.check_flasc_format()


def test_convert_flasc_wide_to_user_wide():
    df_wide_flasc = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    df_wide_user = FlascDataFrame(test_wide_user_dict, channel_name_map=test_channel_name_map)

    pd.testing.assert_frame_equal(df_wide_flasc.convert_to_user_format(), df_wide_user)


def test_convert_flasc_wide_to_user_wide_incomplete():
    # Test incomplete channel name map
    df_wide_flasc_incomplete = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map_incomplete
    )
    df_wide_user_incomplete = FlascDataFrame(
        test_wide_user_dict_incomplete, channel_name_map=test_channel_name_map_incomplete
    )

    pd.testing.assert_frame_equal(
        df_wide_flasc_incomplete.convert_to_user_format(), df_wide_user_incomplete
    )


def test_convert_user_wide_to_flasc_wide():
    df_wide_flasc = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    df_wide_user = FlascDataFrame(test_wide_user_dict, channel_name_map=test_channel_name_map)

    pd.testing.assert_frame_equal(df_wide_user.convert_to_flasc_format(), df_wide_flasc)


def test_convert_user_wide_to_flasc_wide_incomplete():
    # Test incomplete channel name map
    df_wide_flasc_incomplete = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map_incomplete
    )
    df_wide_user_incomplete = FlascDataFrame(
        test_wide_user_dict_incomplete, channel_name_map=test_channel_name_map_incomplete
    )

    pd.testing.assert_frame_equal(
        df_wide_user_incomplete.convert_to_flasc_format(), df_wide_flasc_incomplete
    )


def test_convert_flasc_wide_in_place():
    df_wide_flasc = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    df_wide_user = FlascDataFrame(test_wide_user_dict, channel_name_map=test_channel_name_map)

    df_wide_flasc.convert_to_user_format(inplace=True)
    pd.testing.assert_frame_equal(df_wide_flasc, df_wide_user)


def test_convert_user_wide_in_place():
    df_wide_flasc = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    df_wide_user = FlascDataFrame(test_wide_user_dict, channel_name_map=test_channel_name_map)

    df_wide_user.convert_to_flasc_format(inplace=True)
    pd.testing.assert_frame_equal(df_wide_user, df_wide_flasc)


def test_convert_flasc_wide_back_and_forth():
    df_wide_flasc = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    df_wide_flasc_copy = df_wide_flasc.copy()

    df_wide_flasc.convert_to_user_format(inplace=True)
    df_wide_flasc.convert_to_flasc_format(inplace=True)

    pd.testing.assert_frame_equal(df_wide_flasc, df_wide_flasc_copy)


def test_convert_long_column_names():
    long_col_names = {"variable_column": "VA", "value_column": "VB"}
    df_wide_flasc = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map, long_data_columns=long_col_names
    )
    df_long = df_wide_flasc.convert_to_user_format()

    # Check that df_long has 3 columns named "VA", "VB", and "time"
    assert "VA" in df_long.columns
    assert "VB" in df_long.columns
    assert "time" in df_long.columns


def test_convert_flasc_to_user_long():
    df_wide_flasc = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )
    df_long = FlascDataFrame(
        test_long_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )

    assert_equal_except_row_col_order(df_wide_flasc.convert_to_user_format(), df_long)


def test_convert_flasc_to_user_long_incomplete():
    # Test incomplete channel name map
    df_wide_flasc_incomplete = FlascDataFrame(
        test_wide_dict,
        channel_name_map=test_channel_name_map_incomplete,
        long_data_columns=test_long_columns,
    )
    df_long_incomplete = FlascDataFrame(
        test_long_dict_incomplete,
        channel_name_map=test_channel_name_map_incomplete,
        long_data_columns=test_long_columns,
    )

    assert_equal_except_row_col_order(
        df_wide_flasc_incomplete.convert_to_user_format(), df_long_incomplete
    )


def test_convert_user_long_to_flasc():
    df_wide_flasc = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )
    df_long = FlascDataFrame(
        test_long_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )

    # Note that the column order is different so fix that
    assert_equal_except_row_col_order(df_long.convert_to_flasc_format(), df_wide_flasc)


def test_convert_user_long_to_flasc_incomplete():
    # Test incomplete channel name map
    df_wide_flasc_incomplete = FlascDataFrame(
        test_wide_dict,
        channel_name_map=test_channel_name_map_incomplete,
        long_data_columns=test_long_columns,
    )
    df_long_incomplete = FlascDataFrame(
        test_long_dict_incomplete,
        channel_name_map=test_channel_name_map_incomplete,
        long_data_columns=test_long_columns,
    )

    # Note that the column order is different so fix that
    assert_equal_except_row_col_order(
        df_long_incomplete.convert_to_flasc_format(), df_wide_flasc_incomplete
    )


def test_convert_flasc_long_in_place():
    df_wide_flasc = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )
    df_long = FlascDataFrame(
        test_long_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )

    df_wide_flasc.convert_to_user_format(inplace=True)
    assert_equal_except_row_col_order(df_wide_flasc, df_long)


def test_convert_user_long_in_place():
    df_wide_flasc = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )
    df_long = FlascDataFrame(
        test_long_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )

    df_long.convert_to_flasc_format(inplace=True)
    assert_equal_except_row_col_order(df_long, df_wide_flasc)


def test_convert_flasc_long_back_and_forth():
    df_wide_flasc = FlascDataFrame(
        test_wide_dict, channel_name_map=test_channel_name_map, long_data_columns=test_long_columns
    )
    df_wide_flasc_copy = df_wide_flasc.copy()

    df_wide_flasc.convert_to_user_format(inplace=True)
    df_wide_flasc.convert_to_flasc_format(inplace=True)

    assert_equal_except_row_col_order(df_wide_flasc, df_wide_flasc_copy)


def test_pickle():
    df = FlascDataFrame(test_wide_dict)
    df.channel_name_map = test_channel_name_map
    df.to_pickle("test_pickle.pkl")

    df2 = pd.read_pickle("test_pickle.pkl")
    assert isinstance(df2, FlascDataFrame)
    assert df2.channel_name_map == test_channel_name_map

    os.remove("test_pickle.pkl")


def test_feather():
    df = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    df.to_feather("test_feather.ftr")

    df2 = pd.read_feather("test_feather.ftr")
    # Loaded DataFrame is a pandas DataFrame, not a FlascDataFrame
    assert not isinstance(df2, FlascDataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert not hasattr(df2, "channel_name_map")

    os.remove("test_feather.ftr")


def test_csv():
    df = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    df.to_csv("test_csv.csv")

    df2 = pd.read_csv("test_csv.csv")
    # Loaded DataFrame is a pandas DataFrame, not a FlascDataFrame
    assert not isinstance(df2, FlascDataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert not hasattr(df2, "channel_name_map")

    os.remove("test_csv.csv")


def test_n_turbines():
    # Currently, n_turbines based only on number of pow columns
    df = FlascDataFrame(test_wide_dict, channel_name_map=test_channel_name_map)
    assert df.n_turbines == 2

    # Check n_turbines not valid if not in flasc format
    df.convert_to_user_format(inplace=True)
    with pytest.raises(ValueError):
        df.n_turbines
