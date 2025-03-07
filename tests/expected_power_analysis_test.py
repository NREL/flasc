from itertools import product

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal
from pytest import approx

from flasc.analysis.analysis_input import AnalysisInput
from flasc.analysis.expected_power_analysis import (
    _total_uplift_expected_power_single,
    _total_uplift_expected_power_with_bootstrapping,
    _total_uplift_expected_power_with_standard_error,
    total_uplift_expected_power,
)
from flasc.analysis.expected_power_analysis_utilities import (
    _add_wd_ws_bins,
    _bin_and_group_dataframe_expected_power,
    _compute_covariance,
    _fill_cov_with_var,
    _get_num_points_pair,
    _set_cov_to_zero,
    _synchronize_nulls,
    # _synchronize_cov_nulls_back_to_mean,
    _synchronize_var_nulls_back_to_mean,
)


def load_data():
    # Test the returned energy ratio assuming alternative weightings of the wind speed bins
    df_base = pd.DataFrame(
        {
            "wd": [270, 270.0, 270.0, 270.0, 280.0, 280.0, 280.0, 290.0],
            "ws": [8.0, 8.0, 8.0, 9.0, 8.0, 8.0, 9.0, 8.0],
            "pow_000": [10.0, 20.0, np.nan, 10.0, np.nan, 10.0, np.nan, 10.0],
            "pow_001": [10.0, 20.0, np.nan, 10.0, 10.0, 20.0, np.nan, 10.0],
        }
    )

    df_wake_steering = pd.DataFrame(
        {
            "wd": [270, 270.0, 280.0, 280.0, 280.0, 290.0, 290.0],
            "ws": [8.0, 8.0, 8.0, 8.0, 9.0, 8.0, 8.0],
            "pow_000": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "pow_001": [20.0, 30.0, np.nan, np.nan, 10.0, 10.0, 10.0],
        }
    )

    a_in = AnalysisInput([df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=3)

    return a_in


def load_repeated_data():
    df_base = pd.DataFrame(
        {
            "wd": [270, 270.0, 270.0, 270.0, 280.0, 280.0, 280.0, 290.0],
            "ws": [8.0, 8.0, 8.0, 9.0, 8.0, 8.0, 9.0, 8.0],
            "pow_000": [10.0, 20.0, np.nan, 10.0, np.nan, 10.0, np.nan, 10.0],
            "pow_001": [10.0, 20.0, np.nan, 10.0, 10.0, 20.0, np.nan, 10.0],
        }
    )

    df_wake_steering = pd.DataFrame(
        {
            "wd": [270, 270.0, 280.0, 280.0, 280.0, 290.0, 290.0],
            "ws": [8.0, 8.0, 8.0, 8.0, 9.0, 8.0, 8.0],
            "pow_000": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "pow_001": [20.0, 30.0, np.nan, np.nan, 10.0, 10.0, 10.0],
        }
    )

    # Set df_base to repeat, once with all pow columns -1 and once with
    #  all pow columns +2 to add variance
    df_base_1 = df_base.copy()
    df_base_1["pow_000"] = df_base_1["pow_000"] - 1
    df_base_1["pow_001"] = df_base_1["pow_001"] - 1
    df_base_2 = df_base.copy()
    df_base_2["pow_000"] = df_base_2["pow_000"] + 1
    df_base_2["pow_001"] = df_base_2["pow_001"] + 1
    df_base = pd.concat([df_base, df_base_1, df_base_2], ignore_index=True)

    # Repeat for df_wake_steering
    df_wake_steering_1 = df_wake_steering.copy()
    df_wake_steering_1["pow_000"] = df_wake_steering_1["pow_000"] - 1
    df_wake_steering_1["pow_001"] = df_wake_steering_1["pow_001"] - 1
    df_wake_steering_2 = df_wake_steering.copy()
    df_wake_steering_2["pow_000"] = df_wake_steering_2["pow_000"] + 1
    df_wake_steering_2["pow_001"] = df_wake_steering_2["pow_001"] + 1
    df_wake_steering = pd.concat(
        [df_wake_steering, df_wake_steering_1, df_wake_steering_2], ignore_index=True
    )

    a_in = AnalysisInput([df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=3)

    return a_in


def test_add_wd_ws_bins():
    a_in = load_data()

    df_ = _add_wd_ws_bins(
        df_=a_in.get_df(),
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
    )

    # Sort df_ by wd_bin and ws_bin and df_name
    df_ = df_.sort(["wd_bin", "ws_bin", "df_name"])

    # Assert the first 3 values of ws_bin are 8.0
    np.testing.assert_array_equal(df_["ws_bin"].to_numpy()[:3], np.array([8.0, 8.0, 8.0]))

    # Assert the first 3 values of wd_bin are 270.0
    np.testing.assert_array_equal(df_["wd_bin"].to_numpy()[:3], np.array([270.0, 270.0, 270.0]))

    with pl.Config(tbl_cols=-1):
        print(df_)


def test_bin_and_group_dataframe_expected_power():
    a_in = load_data()

    df_ = _add_wd_ws_bins(
        df_=a_in.get_df(),
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
    )

    df_ = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=["pow_000", "pow_001"],
    )

    # Sort df_ by wd_bin and ws_bin and df_name
    df_ = df_.sort(["wd_bin", "ws_bin", "df_name"])

    # Test the values
    np.testing.assert_array_equal(
        df_["wd_bin"].to_numpy(), np.array([270.0, 270.0, 280.0, 280.0, 290.0, 290.0])
    )
    np.testing.assert_array_equal(
        df_["ws_bin"].to_numpy(), np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    )
    np.testing.assert_array_equal(
        df_["df_name"].to_numpy(),
        np.array(
            [
                "baseline",
                "wake_steering",
                "baseline",
                "wake_steering",
                "baseline",
                "wake_steering",
            ]
        ),
    )
    np.testing.assert_array_equal(
        df_["pow_000_mean"].to_numpy(), np.array([15.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    )
    np.testing.assert_array_equal(
        df_["pow_001_mean"].to_numpy(), np.array([15.0, 25.0, 15.0, np.nan, 10.0, 10.0])
    )

    np.testing.assert_array_equal(df_["pow_000_count"].to_numpy(), np.array([2, 2, 1, 2, 1, 2]))
    np.testing.assert_array_equal(df_["pow_001_count"].to_numpy(), np.array([2, 2, 2, 0, 1, 2]))


def test_synchronize_nulls():
    a_in = load_data()

    df_ = _add_wd_ws_bins(
        df_=a_in.get_df(),
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
    )

    df_ = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=["pow_000", "pow_001"],
    )

    # Sort df_ by wd_bin and ws_bin and df_name
    df_ = df_.sort(["wd_bin", "ws_bin", "df_name"])

    # Synchronize the null values
    df_ = _synchronize_nulls(
        df_,
        sync_cols=[f"{col}_mean" for col in ["pow_000", "pow_001"]],
        uplift_pairs=[["baseline", "wake_steering"]],
    )

    # Assert the nan is copied over
    np.testing.assert_array_equal(
        df_["pow_001_mean"].to_numpy(), np.array([15.0, 25.0, np.nan, np.nan, 10.0, 10.0])
    )

    # Add a hypothetical 3rd df_name with on 290/8 with nans in pow_000 and pow_001
    # to df_
    df_add_row = pl.DataFrame(
        {
            "wd_bin": [290.0],
            "ws_bin": [8.0],
            "df_name": ["hypothetical"],
            "pow_000_mean": [np.nan],
            "pow_001_mean": [np.nan],
            "pow_000_count": [1],
            "pow_001_count": [1],
            "count": [1],
        }
    )

    # Ensure df_add_row has all same types for eac column as df_
    for col in df_.columns:
        df_add_row = df_add_row.with_columns(df_add_row[col].cast(df_[col].dtype))

    # Vstack
    df_ = df_.vstack(df_add_row)

    # Synchronize the null values
    df_ = _synchronize_nulls(
        df_,
        sync_cols=[f"{col}_mean" for col in ["pow_000", "pow_001"]],
        uplift_pairs=[["baseline", "wake_steering"]],
    )

    # Assert the new nan is not copied over
    np.testing.assert_array_equal(
        df_["pow_001_mean"].to_numpy(), np.array([15.0, 25.0, np.nan, np.nan, 10.0, 10.0, np.nan])
    )


def test_synchronize_nulls_with_cov():
    a_in = load_data()

    df_ = _add_wd_ws_bins(
        df_=a_in.get_df(),
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
    )

    # Compute the covariance frame
    test_cols = ["pow_000", "pow_001"]
    bin_cols_with_df_name = ["wd_bin", "ws_bin", "df_name"]
    df_cov = _compute_covariance(
        df_, test_cols=test_cols, bin_cols_with_df_name=bin_cols_with_df_name
    )

    df_bin = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=["pow_000", "pow_001"],
    )

    # Join the covariance dataframe to df_bin
    df_bin = df_bin.join(df_cov, on=bin_cols_with_df_name, how="left")

    # Sort df_ by wd_bin and ws_bin and df_name
    df_bin = df_bin.sort(["wd_bin", "ws_bin", "df_name"])

    # Get the names of all the mean, covariance and num_points columns
    mean_cols = [f"{col}_mean" for col in test_cols]
    cov_cols = [f"cov_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]
    n_cols = [f"count_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]

    with pl.Config(tbl_cols=-1):
        print(df_bin)

    # Synchronize the null values
    df_bin = _synchronize_nulls(
        df_bin,
        sync_cols=mean_cols + cov_cols + n_cols,
        uplift_pairs=[["baseline", "wake_steering"]],
    )

    with pl.Config(tbl_cols=-1):
        print(df_bin)

    # Assert the nan is copied over
    np.testing.assert_array_equal(
        df_bin["cov_pow_000_pow_000"].to_numpy(),
        np.array([50.0, 0.0, np.nan, np.nan, np.nan, np.nan]),
    )


def test_total_uplift_expected_power_single():
    a_in = load_data()

    df_bin, df_sum, uplift_results = _total_uplift_expected_power_single(
        df_=a_in.get_df(),
        test_cols=["pow_000", "pow_001"],
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
    )

    ## Assert df_bin has 6 rows
    assert len(df_bin) == 6

    assert uplift_results["scada_uplift"] == 1.1


def test_total_uplift_expected_power_single_no_nulls():
    a_in = load_data()

    df_bin, df_sum, uplift_results = _total_uplift_expected_power_single(
        df_=a_in.get_df(),
        test_cols=["pow_000", "pow_001"],
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        remove_any_null_turbine_bins=True,
    )

    ## Assert df_bin has only 4 rows
    assert len(df_bin) == 4


def test_total_uplift_expected_power_with_bootstrapping():
    # Set the random seed
    np.random.seed(0)

    a_in = load_data()

    _, _, bootstrap_uplift_result = _total_uplift_expected_power_with_bootstrapping(
        a_in=a_in,
        test_cols=["pow_000", "pow_001"],
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        N=3,
    )

    assert bootstrap_uplift_result["scada_uplift"]["energy_uplift_ctr"] == 1.1
    np.testing.assert_almost_equal(
        bootstrap_uplift_result["scada_uplift"]["energy_uplift_ctr_pc"], 10
    )
    np.testing.assert_almost_equal(
        bootstrap_uplift_result["scada_uplift"]["energy_uplift_lb_pc"],
        (bootstrap_uplift_result["scada_uplift"]["energy_uplift_lb"] - 1) * 100.0,
    )
    np.testing.assert_almost_equal(
        bootstrap_uplift_result["scada_uplift"]["energy_uplift_ub_pc"],
        (bootstrap_uplift_result["scada_uplift"]["energy_uplift_ub"] - 1) * 100.0,
    )


def test_get_num_points_pair():
    test_df = pl.DataFrame(
        {
            "wd_bin": [0, 0, 0, 1, 1, 1],
            "ws_bin": [1, 1, 1, 1, 1, 1],
            "df_name": ["baseline", "baseline", "baseline", "baseline", "baseline", "baseline"],
            "pow_000": np.array([1.0, np.nan, 1.0, 1, 1, 1]),
            "pow_001": np.array([1.0, 1.0, np.nan, 1, np.nan, 1]),
            "pow_002": np.array([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan]),
        }
    )

    df_count = _get_num_points_pair(
        test_df, ["pow_000", "pow_001", "pow_002"], ["wd_bin", "ws_bin", "df_name"]
    )

    with pl.Config(tbl_cols=-1):
        print(df_count)

    df_expected = pl.DataFrame(
        {
            "wd_bin": [0, 1],
            "ws_bin": [1, 1],
            "df_name": ["baseline", "baseline"],  #
            "count": [3, 3],
            "count_pow_000_pow_000": [2, 3],
            "count_pow_000_pow_001": [1, 2],
            "count_pow_000_pow_002": [1, None],
            "count_pow_001_pow_000": [1, 2],
            "count_pow_001_pow_001": [2, 2],
            "count_pow_001_pow_002": [None, None],
            "count_pow_002_pow_000": [1, None],
            "count_pow_002_pow_001": [None, None],
            "count_pow_002_pow_002": [1, None],
        }
    )

    # Test that df_count and df_expected are essentially equal
    assert_frame_equal(
        df_count[["count_pow_000_pow_000"]],
        df_expected[["count_pow_000_pow_000"]],
        check_row_order=False,
        check_dtypes=False,
    )
    assert df_count["count_pow_000_pow_002"][1] is None


def test_compute_covariance():
    test_df = pl.DataFrame(
        {
            "wd_bin": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "ws_bin": np.ones(12),
            "df_name": ["baseline"] * 3
            + ["wake_steering"] * 3
            + ["baseline"] * 3
            + ["wake_steering"] * 3,
            "pow_000": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0]),
            "pow_001": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]),
        }
    )

    df_cov = _compute_covariance(
        df_=test_df,
        test_cols=["pow_000", "pow_001"],
        bin_cols_with_df_name=["wd_bin", "ws_bin", "df_name"],
    )

    df_cov = df_cov.sort(["wd_bin", "ws_bin", "df_name"])

    np.testing.assert_allclose(df_cov["cov_pow_000_pow_001"].to_numpy(), [0, 0, 0, -1])

    # with pl.Config(tbl_cols=-1):
    #     print(test_df)
    #     print(df_cov)


def test_cov_against_var():
    """Test that computing var of one signal is the same as the covariance of the signal."""
    test_df = pl.DataFrame(
        {
            "wd_bin": np.zeros(5),
            "ws_bin": np.ones(5),
            "df_name": ["baseline"] * 5,
            "pow_000": [0, 1, 2, 3, None],
        }
    )

    df_cov = _compute_covariance(
        df_=test_df,
        test_cols=["pow_000"],
        bin_cols_with_df_name=["wd_bin", "ws_bin", "df_name"],
    )

    # Manually compute the variance of the pow_000 column
    man_var = test_df["pow_000"].var()

    np.testing.assert_almost_equal(df_cov["cov_pow_000_pow_000"].to_numpy(), man_var)


def test_fill_cov_with_var_dont_fill_all():
    """Test the fill_cov_null function."""
    test_df = pl.DataFrame(
        {
            "wd_bin": [0, 1],
            "ws_bin": [0, 0],
            "df_name": ["baseline"] * 2,
            "cov_pow_000_pow_000": [4, 4],
            "cov_pow_000_pow_001": [1, None],
            "cov_pow_001_pow_000": [1, 1],
            "cov_pow_001_pow_001": [4, 4],
            "count_pow_000_pow_000": [1, 2],
            "count_pow_000_pow_001": [3, 4],
            "count_pow_001_pow_000": [5, 6],
            "count_pow_001_pow_001": [7, 8],
        }
    )

    expected_df = pl.DataFrame(
        {
            "wd_bin": [0, 1],
            "ws_bin": [0, 0],
            "df_name": ["baseline"] * 2,
            "cov_pow_000_pow_000": [4, 4],
            "cov_pow_000_pow_001": [1, 4],  # Note filled value
            "cov_pow_001_pow_000": [1, 1],
            "cov_pow_001_pow_001": [4, 4],
            "count_pow_000_pow_000": [1, 2],
            "count_pow_000_pow_001": [3, 2],  # Note values are updated here too as min
            "count_pow_001_pow_000": [5, 6],
            "count_pow_001_pow_001": [7, 8],
        }
    )

    filled_df = _fill_cov_with_var(test_df, test_cols=["pow_000", "pow_001"], fill_all=False)

    assert_frame_equal(filled_df, expected_df, check_row_order=False, check_dtypes=False)


def test_fill_cov_with_var_fill_all():
    """Test the fill_cov_null function."""
    test_df = pl.DataFrame(
        {
            "wd_bin": [0, 1],
            "ws_bin": [0, 0],
            "df_name": ["baseline"] * 2,
            "cov_pow_000_pow_000": [4, 4],
            "cov_pow_000_pow_001": [1, None],
            "cov_pow_001_pow_000": [1, 1],
            "cov_pow_001_pow_001": [4, 9],
            "count_pow_000_pow_000": [1, 2],
            "count_pow_000_pow_001": [3, 4],
            "count_pow_001_pow_000": [5, 6],
            "count_pow_001_pow_001": [7, 8],
        }
    )

    expected_df = pl.DataFrame(
        {
            "wd_bin": [0, 1],
            "ws_bin": [0, 0],
            "df_name": ["baseline"] * 2,
            "cov_pow_000_pow_000": [4, 4],
            "cov_pow_000_pow_001": [
                4,
                6,
            ],  # Note filled values
            "cov_pow_001_pow_000": [
                4,
                6,
            ],  # Note filled values (None because count goes below 2)
            "cov_pow_001_pow_001": [4, 9],
            "count_pow_000_pow_000": [1, 2],
            "count_pow_000_pow_001": [1, 2],  # Note values updated here to minimum
            "count_pow_001_pow_000": [1, 2],  # Note values updated here to minimum
            "count_pow_001_pow_001": [7, 8],
        }
    )

    filled_df = _fill_cov_with_var(test_df, test_cols=["pow_000", "pow_001"], fill_all=True)

    with pl.Config(tbl_cols=-1):
        print(test_df)
        print(filled_df)

    assert_frame_equal(filled_df, expected_df, check_row_order=False, check_dtypes=False)


def test_set_cov_to_zero():
    """Test the zero_cov function."""
    test_df = pl.DataFrame(
        {
            "wd_bin": [0, 1],
            "ws_bin": [0, 0],
            "df_name": ["baseline"] * 2,
            "cov_pow_000_pow_000": [4, 4],
            "cov_pow_000_pow_001": [1, None],
            "cov_pow_001_pow_000": [1, 1],
            "cov_pow_001_pow_001": [4, 4],
            "count_pow_000_pow_000": [1, 2],
            "count_pow_000_pow_001": [3, 4],
            "count_pow_001_pow_000": [5, 6],
            "count_pow_001_pow_001": [7, 8],
        }
    )

    expected_df = pl.DataFrame(
        {
            "wd_bin": [0, 1],
            "ws_bin": [0, 0],
            "df_name": ["baseline"] * 2,
            "cov_pow_000_pow_000": [4, 4],
            "cov_pow_000_pow_001": [0, 0],
            "cov_pow_001_pow_000": [0, 0],
            "cov_pow_001_pow_001": [4, 4],
            "count_pow_000_pow_000": [1, 2],
            "count_pow_000_pow_001": [2, 2],
            "count_pow_001_pow_000": [2, 2],
            "count_pow_001_pow_001": [7, 8],
        }
    )

    zero_cov_df = _set_cov_to_zero(test_df, test_cols=["pow_000", "pow_001"])

    assert_frame_equal(zero_cov_df, expected_df, check_row_order=False, check_dtypes=False)


def test_synchronize_var_nulls_back_to_mean():
    test_df_bin = pl.DataFrame(
        {
            "wd_bin": [0, 0, 1, 1],
            "ws_bin": [0, 1, 0, 1],
            "df_name": ["baseline", "baseline", "baseline", "baseline"],
            "pow_000_mean": [1, 2, 3, 4],
            "pow_001_mean": [5, 6, 7, 8],
        }
    )

    test_df_cov = pl.DataFrame(
        {
            "wd_bin": [0, 0, 1, 1],
            "ws_bin": [0, 1, 0, 1],
            "df_name": ["baseline", "baseline", "baseline", "baseline"],
            "cov_pow_000_pow_000": [1, 2, 3, None],
            "cov_pow_000_pow_001": [5, 6, None, 8],
            "cov_pow_001_pow_000": [9, 10, 11, 12],
            "cov_pow_001_pow_001": [13, 14, 15, 16],
        }
    )

    expected_df_bin = pl.DataFrame(
        {
            "wd_bin": [0, 0, 1, 1],
            "ws_bin": [0, 1, 0, 1],
            "df_name": ["baseline", "baseline", "baseline", "baseline"],
            "pow_000_mean": [1, 2, 3, None],
            "pow_001_mean": [5, 6, 7, 8],
        }
    )

    # Join the covariance dataframe to df_bin
    test_df_bin = test_df_bin.join(test_df_cov, on=["wd_bin", "ws_bin", "df_name"], how="left")
    expected_df_bin = expected_df_bin.join(
        test_df_cov, on=["wd_bin", "ws_bin", "df_name"], how="left"
    )

    df_res = _synchronize_var_nulls_back_to_mean(
        df_bin=test_df_bin,
        test_cols=["pow_000", "pow_001"],
    )

    assert_frame_equal(df_res, expected_df_bin, check_row_order=False, check_dtypes=False)


def test_total_uplift_expected_power_with_standard_error():
    a_in = load_repeated_data()

    uplift_results = _total_uplift_expected_power_with_standard_error(
        df_=a_in.get_df(),
        test_cols=["pow_000", "pow_001"],
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        wd_cols=["wd"],
        ws_cols=["ws"],
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
    )

    with pl.Config(tbl_cols=-1):
        print(uplift_results)

    assert uplift_results["scada_uplift"]["energy_uplift_ctr"] == 1.1


def test_center_uplift_identical():
    a_in = load_repeated_data()

    epao_single = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=False,
        N=1,
    )

    epao_boot = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=False,
        N=3,
    )

    epao_standard = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=True,
    )

    epao_standard_fill_with_var = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=True,
        cov_terms="var",
    )

    assert epao_single.uplift_results["scada_uplift"] == 1.1
    assert epao_boot.uplift_results["scada_uplift"]["energy_uplift_ctr"] == 1.1
    assert epao_standard.uplift_results["scada_uplift"]["energy_uplift_ctr"] == 1.1
    assert epao_standard_fill_with_var.uplift_results["scada_uplift"]["energy_uplift_ctr"] == 1.1


def test_uncertain_intervals():
    a_in = load_repeated_data()

    epao_boot = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=False,
        N=10,
        remove_any_null_turbine_bins=False,
    )

    epao_standard_zero = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=True,
        cov_terms="zero",
        remove_any_null_turbine_bins=False,
    )

    epao_standard_var = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=True,
        cov_terms="var",
        remove_any_null_turbine_bins=False,
    )

    print("Bootstrapping")
    print(epao_boot.uplift_results["scada_uplift"]["energy_uplift_ctr"])
    print(epao_boot.uplift_results["scada_uplift"]["energy_uplift_lb"])
    print(epao_boot.uplift_results["scada_uplift"]["energy_uplift_ub"])
    assert (
        epao_boot.uplift_results["scada_uplift"]["energy_uplift_lb"]
        <= epao_boot.uplift_results["scada_uplift"]["energy_uplift_ctr"]
    )
    assert (
        epao_boot.uplift_results["scada_uplift"]["energy_uplift_ub"]
        >= epao_boot.uplift_results["scada_uplift"]["energy_uplift_ctr"]
    )

    print("Standard error zero")
    print(epao_standard_zero.uplift_results["scada_uplift"]["energy_uplift_ctr"])
    print(epao_standard_zero.uplift_results["scada_uplift"]["energy_uplift_lb"])
    print(epao_standard_zero.uplift_results["scada_uplift"]["energy_uplift_ub"])
    assert (
        epao_standard_zero.uplift_results["scada_uplift"]["energy_uplift_lb"]
        <= epao_standard_zero.uplift_results["scada_uplift"]["energy_uplift_ctr"]
    )
    assert (
        epao_standard_zero.uplift_results["scada_uplift"]["energy_uplift_ub"]
        >= epao_standard_zero.uplift_results["scada_uplift"]["energy_uplift_ctr"]
    )

    print("Standard error var")
    print(epao_standard_var.uplift_results["scada_uplift"]["energy_uplift_ctr"])
    print(epao_standard_var.uplift_results["scada_uplift"]["energy_uplift_lb"])
    print(epao_standard_var.uplift_results["scada_uplift"]["energy_uplift_ub"])
    assert (
        epao_standard_var.uplift_results["scada_uplift"]["energy_uplift_lb"]
        <= epao_standard_var.uplift_results["scada_uplift"]["energy_uplift_ctr"]
    )
    assert (
        epao_standard_var.uplift_results["scada_uplift"]["energy_uplift_ub"]
        >= epao_standard_var.uplift_results["scada_uplift"]["energy_uplift_ctr"]
    )


def test_expected_power_var_cov_option():
    """Test uplift center and variance using cov option when dropping rows with missing vars."""
    a_in = load_data()

    epao_standard_cov_or_var = total_uplift_expected_power(
        a_in=a_in,
        uplift_pairs=[("baseline", "wake_steering")],
        uplift_names=["scada_uplift"],
        test_turbines=[0, 1],
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=1.0,
        wd_min=0.5,
        ws_min=0.5,
        use_standard_error=True,
        cov_terms="cov",
    )

    expected_uplift_var = (25 + 100 * (35 / 30) ** 2) / (30**2)

    assert epao_standard_cov_or_var.uplift_results["scada_uplift"]["energy_uplift_ctr"] == 7 / 6
    assert epao_standard_cov_or_var.uplift_results["scada_uplift"]["energy_uplift_var"] == approx(
        expected_uplift_var
    )
