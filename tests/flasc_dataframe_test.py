import os

import numpy as np
import pandas as pd
import pytest
from wind_up.constants import (
    DataColumns,
    RAW_POWER_COL,
    RAW_WINDSPEED_COL,
    RAW_YAWDIR_COL,
    TIMESTAMP_COL,
)

from flasc.flasc_dataframe import FlascDataFrame

test_data_dict = {"time": [0, 10, 20], "a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}

test_name_map = {"a": "AA"}


def test_type():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    assert isinstance(df, FlascDataFrame)

    df2 = df.drop(columns="c")  # Modifies the dataframe, returns a copy
    assert isinstance(df2, FlascDataFrame)

    # Assert df is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)


def test__metadata():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    df._user_format = "long"
    df._in_flasc_format = False
    df2 = df.drop(columns="c")  # Modifies the dataframe, returns a copy
    assert hasattr(df2, "name_map")
    assert df2.name_map == test_name_map
    assert hasattr(df2, "_user_format")
    assert df2._user_format == "long"
    assert hasattr(df2, "_in_flasc_format")
    assert df2._in_flasc_format == True  # Resets, since "_in_flasc_format" not in _metadata.
    # May want to add "_in_flasc_format" to _metadata in future, but this
    # demonstrates functionality


def test_printout():
    df = FlascDataFrame(test_data_dict, name_map=test_name_map)
    df._in_flasc_format = True
    print(df)
    print("\n")
    df._in_flasc_format = False
    print(df)
    print("\n")
    print(df.head())  # In FLASC format, presumably because .head() returns a reinstantiated copy?


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
    df._user_format = "long"  # Should be detected internally
    df.convert_to_user_format(inplace=True)  # Should not pass

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


def test_convert_to_windup_format():
    example_data = [1.1, 2.1, 3.1]
    more_example_data = [11.1, 12.1, 13.1]
    even_more_example_data = [111.1, 112.1, 113.1]
    still_more_example_data = [-1.1, -2.1, -3.1]
    df = pd.DataFrame(
        {
            "time": pd.date_range("2021-01-01", periods=len(example_data), freq="1min"),
            "is_operation_normal_000": [False, True, True],
            "is_operation_normal_001": [True, True, False],
            "pow_000": example_data,
            "pow_001": example_data,
            "ws_000": more_example_data,
            "ws_001": more_example_data,
            "wd_000": even_more_example_data,
            "wd_001": still_more_example_data,
            "pitch_000": still_more_example_data,
            "pitch_001": even_more_example_data,
        }
    )
    windup_df = FlascDataFrame(df).convert_to_windup_format()
    assert isinstance(windup_df, pd.DataFrame)
    assert windup_df.index.name == TIMESTAMP_COL
    assert DataColumns.turbine_name in windup_df.columns
    assert DataColumns.active_power_mean in windup_df.columns
    assert windup_df[DataColumns.turbine_name].to_list() == ["000"] * len(example_data) + [
        "001"
    ] * len(example_data)
    assert windup_df[DataColumns.active_power_mean].to_list() == example_data * 2
    assert windup_df[DataColumns.wind_speed_mean].to_list() == more_example_data * 2
    assert windup_df[DataColumns.yaw_angle_mean].to_list() == [
        *even_more_example_data,
        *still_more_example_data,
    ]
    assert windup_df["pitch"].to_list() == [*still_more_example_data, *even_more_example_data]
    assert windup_df[DataColumns.pitch_angle_mean].to_list() == [0] * 2 * len(example_data)
    assert windup_df[DataColumns.gen_rpm_mean].to_list() == [1000] * 2 * len(example_data)
    assert windup_df[DataColumns.shutdown_duration].to_list() == [0] * 2 * len(example_data)
    assert windup_df[RAW_POWER_COL].equals(windup_df[DataColumns.active_power_mean])
    windup_df_turbine_names = FlascDataFrame(df).convert_to_windup_format(
        turbine_names=["T1", "T2"]
    )
    assert windup_df_turbine_names[DataColumns.turbine_name].to_list() == ["T1"] * len(
        example_data
    ) + ["T2"] * len(example_data)
    assert (
        windup_df_turbine_names[windup_df_turbine_names[DataColumns.turbine_name] == "T2"][
            DataColumns.yaw_angle_mean
        ].to_list()
        == still_more_example_data
    )
    windup_df_filt = FlascDataFrame(df).convert_to_windup_format(
        normal_operation_col="is_operation_normal"
    )
    assert windup_df_filt[RAW_POWER_COL].equals(windup_df[RAW_POWER_COL])
    assert windup_df_filt[RAW_WINDSPEED_COL].equals(windup_df[RAW_WINDSPEED_COL])
    assert windup_df_filt[RAW_YAWDIR_COL].equals(windup_df[RAW_YAWDIR_COL])
    assert windup_df_filt[DataColumns.active_power_mean].equals(
        pd.DataFrame(
            {"expected": [np.nan, *(example_data * 2)[1:-1], np.nan]}, index=windup_df.index
        )["expected"]
    )
    windup_df_with_real_pitch = FlascDataFrame(df).convert_to_windup_format(
        normal_operation_col="is_operation_normal", pitchangle_col="pitch"
    )
    assert windup_df_with_real_pitch[DataColumns.pitch_angle_mean].equals(
        pd.DataFrame(
            {
                "expected": [
                    np.nan,
                    *[*still_more_example_data, *even_more_example_data][1:-1],
                    np.nan,
                ]
            },
            index=windup_df.index,
        )["expected"]
    )
