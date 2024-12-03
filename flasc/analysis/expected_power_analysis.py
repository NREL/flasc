"""Analyze SCADA data using expected power methods."""

import warnings
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from flasc.analysis.analysis_input import AnalysisInput
from flasc.analysis.expected_power_analysis_output import ExpectedPowerAnalysisOutput
from flasc.analysis.expected_power_analysis_utilities import (
    _add_wd_ws_bins,
    _bin_and_group_dataframe_expected_power,
    _compute_covariance,
    _fill_cov_with_var,
    _null_and_sync_covariance,
    _set_cov_to_zero,
    _synchronize_nulls,
    _synchronize_var_nulls_back_to_mean,
)
from flasc.data_processing.dataframe_manipulations import df_reduce_precision
from flasc.logging_manager import LoggingManager
from flasc.utilities.energy_ratio_utilities import add_bin_weights

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def _total_uplift_expected_power_single(
    df_: pl.DataFrame,
    test_cols: List[str],
    wd_cols: List[str],
    ws_cols: List[str],
    uplift_pairs: List[Tuple[str, str]],
    uplift_names: List[str],
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: List[str] = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq_pl: pl.DataFrame = None,
    remove_any_null_turbine_bins: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
    """Calculate the total uplift in expected power for a single run.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        test_cols (List[str]): A list of column names to calculate uplift over
        wd_cols (List[str]): A list of column names for wind direction
        ws_cols (List[str]): A list of column names for wind speed
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare
        uplift_names (List[str]): A list of names for the uplift results
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method to weight the bins. Defaults to "min".
        df_freq_pl (pl.DataFrame): A polars dataframe with the frequency of each bin.
            Defaults to None.
        remove_any_null_turbine_bins (bool): When computing farm power, remove any bins where
            and of the test turbines is null.  Defaults to False.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]: A tuple containing the binned
            dataframe, the summed dataframe, and the uplift results.
    """
    # Get the bin cols without df_name
    bin_cols_without_df_name = [c for c in bin_cols_in if c != "df_name"]

    # Add the wd and ws bins
    df_ = _add_wd_ws_bins(
        df_,
        wd_cols=wd_cols,
        ws_cols=ws_cols,
        wd_step=wd_step,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_step=ws_step,
        ws_min=ws_min,
        ws_max=ws_max,
    )

    # Bin and group the dataframe
    df_bin = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=test_cols,
        bin_cols_without_df_name=bin_cols_without_df_name,
    )

    # Synchronize the null values
    df_bin = _synchronize_nulls(
        df_bin=df_bin,
        sync_cols=[f"{col}_mean" for col in test_cols],
        uplift_pairs=uplift_pairs,
        bin_cols_without_df_name=bin_cols_without_df_name,
    )

    # Remove rows where any of the test columns are null (if remove_any_null_turbine_bins is True)
    if remove_any_null_turbine_bins:
        df_bin = df_bin.filter(
            pl.all_horizontal([pl.col(f"{c}_mean").is_not_null() for c in test_cols])
        )

    # Compute the farm power
    df_bin = df_bin.with_columns(pow_farm=pl.sum_horizontal([f"{c}_mean" for c in test_cols]))

    # Determine the weighting of the ws/wd bins
    df_bin, df_freq_pl = add_bin_weights(df_bin, df_freq_pl, bin_cols_without_df_name, weight_by)

    # Normalize the weight column over the values in df_name column
    df_bin = df_bin.with_columns(
        weight=pl.col("weight") / pl.col("weight").sum().over("df_name")
    ).with_columns(weighted_power=pl.col("pow_farm") * pl.col("weight"))

    df_sum = df_bin.group_by("df_name").sum()

    # Compute the uplift for each uplift_pair
    uplift_results = {}
    for uplift_pair, uplift_name in zip(uplift_pairs, uplift_names):
        # Get the ratio of weighted_power
        uplift_results[uplift_name] = (
            df_sum.filter(pl.col("df_name") == uplift_pair[1])["weighted_power"].to_numpy()[0]
            / df_sum.filter(pl.col("df_name") == uplift_pair[0])["weighted_power"].to_numpy()[0]
        )

    return df_bin, df_sum, uplift_results


def _total_uplift_expected_power_with_bootstrapping(
    a_in: AnalysisInput,
    test_cols: List[str],
    wd_cols: List[str],
    ws_cols: List[str],
    uplift_pairs: List[Tuple[str, str]],
    uplift_names: List[str],
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: List[str] = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq_pl: pl.DataFrame = None,
    remove_any_null_turbine_bins: bool = False,
    N: int = 1,
    percentiles: List[float] = [2.5, 97.5],
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Dict[str, float]]]:
    """Calculate the total uplift in expected power using bootstrapping.

    Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze
        test_cols (List[str]): A list of column names to calculate the uplift over
        wd_cols (List[str]): A list of column names for wind direction
        ws_cols (List[str]): A list of column names for wind speed
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare
        uplift_names (List[str]): A list of names for the uplift results
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method to weight the bins. Defaults to "min".
        df_freq_pl (pl.DataFrame): A polars dataframe with the frequency of each bin.
            Defaults to None.
        remove_any_null_turbine_bins (bool): When computing farm power, remove any bins where
            and of the test turbines is null.  Defaults to False.
        N (int): The number of bootstrap samples. Defaults to 1.
        percentiles (List[float]): The percentiles to calculate for the bootstrap samples.
            Defaults to [2.5, 97.5].

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Dict[str, float]]]: A tuple containing the
             binned dataframe, the summed dataframe, and the uplift results.
    """
    uplift_single_outs = [
        _total_uplift_expected_power_single(
            a_in.resample_energy_table(perform_resample=(i != 0)),
            test_cols=test_cols,
            wd_cols=wd_cols,
            ws_cols=ws_cols,
            uplift_pairs=uplift_pairs,
            uplift_names=uplift_names,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            bin_cols_in=bin_cols_in,
            weight_by=weight_by,
            df_freq_pl=df_freq_pl,
            remove_any_null_turbine_bins=remove_any_null_turbine_bins,
        )
        for i in range(N)
    ]

    # Add in the statistics
    bootstrap_uplift_result = {}

    for uplift_name in uplift_names:
        uplift_values = np.zeros(N)

        for i in range(N):
            uplift_values[i] = uplift_single_outs[i][2][uplift_name]

        uplift_central = uplift_values[0]
        uplift_lb = np.quantile(uplift_values, percentiles[0] / 100)
        uplift_ub = np.quantile(uplift_values, percentiles[1] / 100)

        bootstrap_uplift_result[uplift_name] = {
            "energy_uplift_ctr": uplift_central,
            "energy_uplift_lb": uplift_lb,
            "energy_uplift_ub": uplift_ub,
            "energy_uplift_ctr_pc": (uplift_central - 1) * 100,
            "energy_uplift_lb_pc": (uplift_lb - 1) * 100,
            "energy_uplift_ub_pc": (uplift_ub - 1) * 100,
        }

    return uplift_single_outs[0][0], uplift_single_outs[0][1], bootstrap_uplift_result


def _total_uplift_expected_power_with_standard_error(
    df_: pl.DataFrame,
    test_cols: List[str],
    wd_cols: List[str],
    ws_cols: List[str],
    uplift_pairs: List[Tuple[str, str]],
    uplift_names: List[str],
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: List[str] = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq_pl: pl.DataFrame = None,
    percentiles: List[float] = [2.5, 97.5],
    remove_any_null_turbine_bins: bool = False,
    set_cov_to_zero_or_var: str = "zero",
    use_cov_when_available: bool = False,
    # variance_only: bool = False,
    # fill_cov_with_var: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Calculate the total uplift in expected power with standard error.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        test_cols (List[str]): A list of column names to calculate the uplift of
        wd_cols (List[str]): A list of column names for wind direction
        ws_cols (List[str]): A list of column names for wind speed
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the
            df_name values to compare
        uplift_names (List[str]): A list of names for the uplift results
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method to weight the bins. Defaults to "min".
        df_freq_pl (pl.DataFrame): A polars dataframe with the frequency of each bin.
            Defaults to None.
        percentiles (List[float]): The percentiles to for confidence intervals.
            Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): When computing farm power, remove any bins where
            and of the test turbines is null.  Defaults to False.
        set_cov_to_zero_or_var (str): Set the covariance to zero or product of variances.
            Can be "zero" or "var". Defaults to "zero".
        use_cov_when_available (bool): Use the covariance terms when available. If True,
            set_cov_to_zero_or_var must be 'var'.  Defaults to False.


    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the uplift results with standard error.
    """
    # Get the bin cols without df_name
    bin_cols_without_df_name = [c for c in bin_cols_in if c != "df_name"]
    bin_cols_with_df_name = bin_cols_in + ["df_name"]

    # Add the wd and ws bins
    df_ = _add_wd_ws_bins(
        df_,
        wd_cols=wd_cols,
        ws_cols=ws_cols,
        wd_step=wd_step,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_step=ws_step,
        ws_min=ws_min,
        ws_max=ws_max,
    )

    # with pl.Config(tbl_cols=-1):
    #     print(df_)

    # Compute the covariance frame
    df_cov = _compute_covariance(
        df_, test_cols=test_cols, bin_cols_with_df_name=bin_cols_with_df_name
    )

    with pl.Config(tbl_cols=-1):
        print(df_cov)

    # In current version of code, covarariances are either set to 0 or set to
    # product of variances
    if set_cov_to_zero_or_var == "zero":
        if use_cov_when_available:
            raise ValueError(
                "use_cov_when_available cannot be True when set_cov_to_zero_or_var is 'zero'"
            )
        else:
            df_cov = _set_cov_to_zero(df_cov, test_cols=test_cols)
    elif set_cov_to_zero_or_var == "var":
        if use_cov_when_available:
            df_cov = _fill_cov_with_var(df_cov, test_cols=test_cols, fill_all=False)
        else:
            df_cov = _fill_cov_with_var(df_cov, test_cols=test_cols, fill_all=True)
    else:
        raise ValueError(
            f"set_cov_to_zero_or_var must be 'zero' or 'var', not {set_cov_to_zero_or_var}"
        )

    # with pl.Config(tbl_cols=-1):
    #     print(df_cov)

    # # If filling missing covariance terms, do it now
    # if fill_cov_with_var:
    #     df_cov = _fill_cov_with_var(df_cov, test_cols=test_cols)

    # # If only using the variance, zero out the covariance terms
    # if variance_only:
    #     df_cov = _set_cov_to_zero(df_cov, test_cols=test_cols)

    # Apply Null values to covariance and sync across uplift pairs
    df_cov = _null_and_sync_covariance(
        df_cov=df_cov,
        test_cols=test_cols,
        uplift_pairs=uplift_pairs,
        bin_cols_without_df_name=bin_cols_without_df_name,
    )

    # with pl.Config(tbl_cols=-1):
    #     print(df_cov)

    # Bin and group the dataframe
    df_bin = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=test_cols,
        bin_cols_without_df_name=bin_cols_without_df_name,
    )

    with pl.Config(tbl_cols=-1):
        print(df_bin)

    # Join the covariance dataframe to df_bin
    df_bin = df_bin.join(df_cov, on=bin_cols_with_df_name, how="left")

    # DROPPING THIS AS REDUNDANT TO COPYING BACK NULL VARIANCE TO MEAN
    # # Synchronize any null values in the covariance back to df_bin
    # df_bin = _synchronize_cov_nulls_back_to_mean(df_bin=df_bin, test_cols=test_cols)

    # Set to null any mean values associated with null variance by row/turbine
    df_bin = _synchronize_var_nulls_back_to_mean(df_bin=df_bin, test_cols=test_cols)

    # Synchronize the null values in the mean columns across uplift pairs
    df_bin = _synchronize_nulls(
        df_bin=df_bin,
        sync_cols=[f"{col}_mean" for col in test_cols],
        uplift_pairs=uplift_pairs,
        bin_cols_without_df_name=bin_cols_without_df_name,
    )

    # Remove rows where any of the test columns are null (if remove_any_null_turbine_bins is True)
    if remove_any_null_turbine_bins:
        df_bin = df_bin.filter(
            pl.all_horizontal([pl.col(f"{c}_mean").is_not_null() for c in test_cols])
        )

    # Remove rows where all of the test_cols are null
    df_bin = df_bin.filter(
        pl.any_horizontal([pl.col(f"{c}_mean").is_not_null() for c in test_cols])
    )

    with pl.Config(tbl_cols=-1):
        print(df_bin)

    # with pl.Config(tbl_cols=-1):
    #     print(df_bin)

    # Get the names of all the covariance and num_points columns
    cov_cols = [f"cov_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]
    n_cols = [f"count_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]

    # with pl.Config(tbl_cols=-1):
    #     print(df_bin)

    # Compute the farm power
    df_bin = df_bin.with_columns(pow_farm=pl.sum_horizontal([f"{c}_mean" for c in test_cols]))

    # Computed the expected farm power variance as the sum of cov_cols / n_cols
    # This is equation 4.16
    df_bin = df_bin.with_columns(
        pow_farm_var=pl.sum_horizontal(
            [pl.col(cov) / pl.col(n) for cov, n in zip(cov_cols, n_cols)]
        )
    )

    # # If any of the cov_cols are null, set pow_farm_var to null
    # df_bin = df_bin.with_columns(
    #     pl.when(pl.all_horizontal([pl.col(c).is_not_null() for c in cov_cols]))
    #     .then(pl.col("pow_farm_var"))
    #     .otherwise(None)
    #     .alias("pow_farm_var")
    # )

    # with pl.Config(tbl_cols=-1):
    #     print(df_bin)

    # Determine the weighting of the ws/wd bins
    df_bin, df_freq_pl = add_bin_weights(df_bin, df_freq_pl, bin_cols_without_df_name, weight_by)

    # Normalize the weight column over the values in df_name column and compute
    # the weighted farm power and weighted var per bin
    df_bin = df_bin.with_columns(weight=pl.col("weight") / pl.col("weight").sum().over("df_name"))

    # Add the weighted farm power column
    df_bin = df_bin.with_columns(weighted_farm_power=pl.col("pow_farm") * pl.col("weight"))
    # .with_columns(weighted_power=pl.col("pow_farm") * pl.col("weight"),
    # weighted_power_var=pl.col("pow_farm_var") * pl.col("weight")**2)

    # Now compute uplifts
    uplift_results = {}
    for uplift_pair, uplift_name in zip(uplift_pairs, uplift_names):
        # Subset the dataframe to the uplift_pair in consideration
        df_sub = df_bin.filter(pl.col("df_name").is_in(uplift_pair))

        # Limit the columns to bin_cols_without_df_name, df_name, weight, pow_farm, pow_farm_var
        df_sub = df_sub.select(
            bin_cols_without_df_name
            + ["df_name", "weight", "weighted_farm_power", "pow_farm", "pow_farm_var"]
        )

        # Keep bin_cols_without_df_name, weight as the index, pivot df_name across pow_farm,
        # pow_farm_var
        df_sub = df_sub.pivot(
            on="df_name",
            index=bin_cols_without_df_name,  # + ["weight"],
        )

        # Assign the weight column to be the mean of weight_[uplift_pair[0]]
        # and weight_[uplift_pair[1]]
        df_sub = df_sub.with_columns(
            weight=(pl.col(f"weight_{uplift_pair[0]}") + pl.col(f"weight_{uplift_pair[1]}")) / 2
        )

        # Remove the weight_pair columns
        df_sub = df_sub.drop([f"weight_{uplift_pair[0]}", f"weight_{uplift_pair[1]}"])

        with pl.Config(tbl_cols=-1):
            print(df_sub)

        # Compute the expected power ratio per bin
        df_sub = df_sub.with_columns(
            expected_power_ratio=pl.col(f"pow_farm_{uplift_pair[1]}")
            / pl.col(f"pow_farm_{uplift_pair[0]}")
        )

        # # Compute the weighted expected power ratio
        # df_sub = df_sub.with_columns(
        #     weighted_expected_power_ratio=pl.col("expected_power_ratio") * pl.col("weight")
        # )

        # Compute the total expected power ratio (note this is not computed by
        #  combining the expected power ratios)
        farm_power_0 = df_sub[f"weighted_farm_power_{uplift_pair[0]}"].sum()
        farm_power_1 = df_sub[f"weighted_farm_power_{uplift_pair[1]}"].sum()
        total_expected_power_ratio = farm_power_1 / farm_power_0

        # Compute the variance of the expected power ratio per bin
        # This is equation 4.22
        df_sub = df_sub.with_columns(
            expected_power_ratio_var=(
                (
                    pl.col(f"pow_farm_var_{uplift_pair[0]}") * pl.col("expected_power_ratio") ** 2
                    + pl.col(f"pow_farm_var_{uplift_pair[1]}")
                )
                / pl.col(f"pow_farm_{uplift_pair[0]}") ** 2
            )
        )

        # Compute the weighted variance of the expected power ratio
        # This is equation 4.29 pre-summation
        df_sub = df_sub.with_columns(
            weighted_expected_power_ratio_var=(
                (
                    pl.col(f"pow_farm_var_{uplift_pair[1]}") * pl.col("weight") ** 2
                    + pl.col(f"pow_farm_var_{uplift_pair[0]}")
                    * total_expected_power_ratio**2
                    * pl.col("weight") ** 2
                )
                / farm_power_0**2
            )
        )

        with pl.Config(tbl_cols=-1):
            print(df_sub)

        # The total uplift is the sum of the weighted expected power ratio and the weighted
        #  variance of the expected power ratio
        result_dict = {}
        result_dict["energy_uplift_ctr"] = total_expected_power_ratio
        result_dict["energy_uplift_var"] = df_sub["weighted_expected_power_ratio_var"].sum()

        # Add the confidence intervals
        z_value = norm.ppf(percentiles[1] / 100.0)
        result_dict["energy_uplift_lb"] = result_dict["energy_uplift_ctr"] - z_value * np.sqrt(
            result_dict["energy_uplift_var"]
        )
        result_dict["energy_uplift_ub"] = result_dict["energy_uplift_ctr"] + z_value * np.sqrt(
            result_dict["energy_uplift_var"]
        )

        result_dict["df"] = df_sub

        uplift_results[uplift_name] = result_dict

    return uplift_results


def total_uplift_expected_power(
    a_in: AnalysisInput,
    uplift_pairs: List[Tuple[str, str]],
    uplift_names: List[str],
    test_turbines: List[int],
    wd_turbines: List[int] = None,
    ws_turbines: List[int] = None,
    use_predefined_wd: bool = False,
    use_predefined_ws: bool = False,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: List[str] = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq: pd.DataFrame = None,
    use_standard_error: bool = True,
    N: int = 1,
    percentiles: List[float] = [2.5, 97.5],
    remove_any_null_turbine_bins: bool = False,
    set_cov_to_zero_or_var: str = "zero",
    use_cov_when_available: bool = False,
    # variance_only: bool = False,
    # fill_cov_with_var: bool = False,
) -> ExpectedPowerAnalysisOutput:
    """Calculate the total uplift in energy production using expected power methods.

    Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare
        uplift_names (List[str]): A list of names for the uplift results
        test_turbines (List[int]): A list of turbine indices to test
        wd_turbines (List[int]): A list of turbine indices for wind direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for wind speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method to weight the bins. Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin. Defaults to None.
        use_standard_error (bool): Use standard error for the uplift calculation. Defaults to True.
        N (int): The number of bootstrap samples. Defaults to 1.
        percentiles (List[float]): The percentiles to calculate for the bootstrap samples.
            Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): When computing farm power, remove any bins where
            and of the test turbines is null.  Defaults to False.
        set_cov_to_zero_or_var (str): Set the covariance to zero or product of variances.
            Can be "zero" or "var". Defaults to "zero".
        use_cov_when_available (bool): Use the covariance terms when available. If True,
            set_cov_to_zero_or_var must be 'var'.  Defaults to False.

    Returns:
        ExpectedPowerAnalysisOutput: An object containing the uplift results and
            analysis parameters.
    """
    # Get the polars dataframe from within the a_in
    df_ = a_in.get_df()

    # Confirm that test_turbines is a list of ints or a numpy array of ints
    if not isinstance(test_turbines, list) and not isinstance(test_turbines, np.ndarray):
        raise ValueError("test_turbines must be a list or numpy array of ints")

    # Confirm that test_turbines is not empty
    if len(test_turbines) == 0:
        raise ValueError("test_turbines cannot be empty")

    # If use_predefined_wd is True, df_ must have a column named 'wd'
    if use_predefined_wd:
        if "wd" not in df_.columns:
            raise ValueError("df_ must have a column named wd when use_predefined_wd is True")
        # If wd_turbines supplied, warn user that it will be ignored
        if wd_turbines is not None:
            warnings.warn("wd_turbines will be ignored when use_predefined_wd is True")
    else:
        # wd_turbine must be supplied
        if wd_turbines is None:
            raise ValueError("wd_turbines must be supplied when use_predefined_wd is False")

    # If use_predefined_ws is True, df_ must have a column named 'ws'
    if use_predefined_ws:
        if "ws" not in df_.columns:
            raise ValueError("df_ must have a column named ws when use_predefined_ws is True")
        # If ws_turbines supplied, warn user that it will be ignored
        if ws_turbines is not None:
            warnings.warn("ws_turbines will be ignored when use_predefined_ws is True")
    else:
        # ws_turbine must be supplied
        if ws_turbines is None:
            raise ValueError("ws_turbines must be supplied when use_predefined_ws is False")

    # Confirm the weight_by argument is valid
    if weight_by not in ["min", "sum"]:
        raise ValueError('weight_by must be one of "min", or "sum"')

    # Confirm df_freq contains ws, wd and freq_val
    if df_freq is not None:
        if (
            ("ws" not in df_freq.columns)
            or ("wd" not in df_freq.columns)
            or ("freq_val" not in df_freq.columns)
        ):
            raise ValueError("df_freq must have columns ws, wd and freq_val")

        # If df_freq is provided, confirm is consistent with ws/wd min max and
        # prepare a polars table of weights
        # Convert to polars dataframe
        df_freq_pl = pl.from_pandas(df_reduce_precision(df_freq, allow_convert_to_integer=False))

        # Rename the columns
        df_freq_pl = df_freq_pl.rename({"ws": "ws_bin", "wd": "wd_bin", "freq_val": "weight"})

    else:
        df_freq_pl = None

    # Check that if use_standard_error is True, than N = 1
    if use_standard_error and N != 1:
        raise ValueError("N must be 1 when use_standard_error is True")

    # # Raise an error if both variance_only and fill_cov_with_var are True
    # May need to reinclude in the future but disabled for now
    # if variance_only and fill_cov_with_var:
    #     raise ValueError("variance_only and fill_cov_with_var cannot both be True")

    # Set up the column names for the wind speed and test cols

    if not use_predefined_ws:
        ws_cols = [f"ws_{i:03d}" for i in ws_turbines]
    else:
        ws_cols = ["ws"]

    if not use_predefined_wd:
        wd_cols = [f"wd_{i:03d}" for i in wd_turbines]
    else:
        wd_cols = ["wd"]

    # Convert the numbered arrays to appropriate column names
    test_cols = [f"pow_{i:03d}" for i in test_turbines]

    # If N = 1 AND use_standard_error if false, then use the single method
    if N == 1 and not use_standard_error:
        df_bin, df_sum, uplift_results = _total_uplift_expected_power_single(
            a_in.get_df(),
            test_cols=test_cols,
            wd_cols=wd_cols,
            ws_cols=ws_cols,
            uplift_pairs=uplift_pairs,
            uplift_names=uplift_names,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            bin_cols_in=bin_cols_in,
            weight_by=weight_by,
            df_freq_pl=df_freq_pl,
            remove_any_null_turbine_bins=remove_any_null_turbine_bins,
        )
    elif N > 1:
        df_bin, df_sum, uplift_results = _total_uplift_expected_power_with_bootstrapping(
            a_in,
            test_cols=test_cols,
            wd_cols=wd_cols,
            ws_cols=ws_cols,
            uplift_pairs=uplift_pairs,
            uplift_names=uplift_names,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            bin_cols_in=bin_cols_in,
            weight_by=weight_by,
            df_freq_pl=df_freq_pl,
            remove_any_null_turbine_bins=remove_any_null_turbine_bins,
            N=N,
            percentiles=percentiles,
        )
    elif use_standard_error:
        uplift_results = _total_uplift_expected_power_with_standard_error(
            a_in.get_df(),
            test_cols=test_cols,
            wd_cols=wd_cols,
            ws_cols=ws_cols,
            uplift_pairs=uplift_pairs,
            uplift_names=uplift_names,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            bin_cols_in=bin_cols_in,
            weight_by=weight_by,
            df_freq_pl=df_freq_pl,
            percentiles=percentiles,
            remove_any_null_turbine_bins=remove_any_null_turbine_bins,
            set_cov_to_zero_or_var=set_cov_to_zero_or_var,
            use_cov_when_available=use_cov_when_available,
        )

    # Create the output object
    epao = ExpectedPowerAnalysisOutput(
        uplift_results=uplift_results,
        a_in=a_in,
        test_turbines=test_turbines,
        wd_turbines=wd_turbines,
        ws_turbines=ws_turbines,
        use_predefined_wd=use_predefined_wd,
        use_predefined_ws=use_predefined_ws,
        wd_step=wd_step,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_step=ws_step,
        ws_min=ws_min,
        ws_max=ws_max,
        bin_cols_in=bin_cols_in,
        weight_by=weight_by,
        df_freq=df_freq,
        uplift_pairs=uplift_pairs,
        uplift_names=uplift_names,
        use_standard_error=use_standard_error,
        N=N,
        percentiles=percentiles,
        remove_any_null_turbine_bins=remove_any_null_turbine_bins,
        set_cov_to_zero_or_var=set_cov_to_zero_or_var,
        use_cov_when_available=use_cov_when_available,
    )

    # Return the object
    return epao
