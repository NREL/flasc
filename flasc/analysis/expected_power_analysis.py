"""Analyze SCADA data using expected power methods."""

# This module contains functions for computing the total wind farm energy uplift between two modes
# of control (e.g., wind farm control and baseline control) using methods described in the report:
#
# Kanev, S., "AWC Validation Methodology,"" TNO 2020 R11300, Tech. rep., TNO, Petten,
# The Netherlands, 2020. https://resolver.tno.nl/uuid:fdae4c94-fbcc-4337-b49f-5a39c93ef2cf.
#
# Specifically, this module computes the total uplift along with the confidence interval of the
# total uplift estimate by implementing Equations 4.11 - 4.29 in the abovementioned report. To
# determine total energy for the two control modes being compared, the expected wind farm power
# is computed for each wind direction/wind speed bin. This is done by summing the expected power
# of each individual turbine for the bin. Note that by computing expected power at the turbine
# level before summing, the method does not require that all test turbines are operating normally
# at each timestamp. Therefore, fewer timestamps need to be discarded than for total uplift methods
# that require all test turbines to be operating normally at each timestamp, such as the energy
# ratio-based approach. Total wind farm energy is then computed by summing the expected farm power
# values weighted by their frequencies of occurrence over all wind condition bins. However, because
# the test turbine power values in this method are not normalized by the power of reference
# turbines, the method may be more sensitive to wind speed variations within bins and other
# atmospheric conditions aside from wind speed or direction that are not controlled for.
#
# The module provides two approaches for quantifying uncertainty in the total uplift. First,
# bootstrapping can be used to estimate uncertainty by randomly resampling the input data with
# replacement and computing the resulting uplift for many iterations. The second option computes
# uncertainty in the total uplift following the approach in the abovementioned TNO report by
# propagating the standard errors of the expected wind farm power in each wind condition bin for
# the two control modes. Benefits of the method of propagating standard errors include an analytic
# expression for calculating uncertainty, rather than replying on the empirical bootstrapping
# approach, and higher computational efficiency compared to bootstrapping, which relies on
# computing the uplift for many different iterations. Some drawbacks of the method include the need
# to linearize the formula for total uplift to permit the standard error to be calculated,
# resulting in an approximation of the uncertainty, as well as challenges with computing the
# required variances and covariances of wind turbine power in bins with very little data.

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
        test_cols (List[str]): A list of column names to include when computing total uplift.
        wd_cols (List[str]): A list of column names for determining the reference wind direction.
        ws_cols (List[str]): A list of column names for determining the reference wind speed.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name values to
            compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq_pl (pl.DataFrame): A polars dataframe with the frequency of each bin.
            Defaults to None.
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.

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
    """Calculate the total uplift in expected power using bootstrapping to quantify uncertainty.

    Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze.
        test_cols (List[str]): A list of column names to include when computing total uplift.
        wd_cols (List[str]): A list of column names for determining the reference wind direction.
        ws_cols (List[str]): A list of column names for determining the reference wind speed.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name values to
            compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq_pl (pl.DataFrame): A polars dataframe with the frequency of each bin.
            Defaults to None.
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        N (int): The number of bootstrap samples. Defaults to 1.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].

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
    cov_terms: str = "zero",
) -> Dict[str, Dict[str, float]]:
    """Calculate total uplift by propagating standard errors.

    Calculate the total uplift in expected power with uncertainty quantified by propagating
    bin-level standard errors.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        test_cols (List[str]): A list of column names to include when computing total uplift.
        wd_cols (List[str]): A list of column names for determining the reference wind direction.
        ws_cols (List[str]): A list of column names for determining the reference wind speed.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name values to
            compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq_pl (pl.DataFrame): A polars dataframe with the frequency of each bin.
            Defaults to None.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        cov_terms (str): The approach for determining the power covariance terms betweens pairs of
            test turbines whe computing uncertainty. Can be "zero", "var" or "cov". If "zero" all
            covariance terms are set to zero. If "var" all covariance terms are set to the product
            of the square root of the variances of each individual turbine. If "cov" the computed
            covariance terms are used as is when enough data are present in a bin, with missing
            terms set to the product of the square root of the individual turbine variances.
            Defaults to "zero".

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

    # Compute the covariance frame
    df_cov = _compute_covariance(
        df_, test_cols=test_cols, bin_cols_with_df_name=bin_cols_with_df_name
    )

    # There are three options for the covariance terms used to compute uncertainty.
    # 1. Zero out all covariance terms ("zero"), leave only variance
    # 2. Fill all covariance terms with the product of the variances ("var")
    # 3. Use the covariance terms as is, with missing terms filled with
    #    the product of the variances ("cov")
    if cov_terms == "zero":
        df_cov = _set_cov_to_zero(df_cov, test_cols=test_cols)
    elif cov_terms == "var":
        df_cov = _fill_cov_with_var(df_cov, test_cols=test_cols, fill_all=True)
    elif cov_terms == "cov":
        df_cov = _fill_cov_with_var(df_cov, test_cols=test_cols, fill_all=False)
    else:
        raise ValueError(f"cov_terms must be 'zero', 'var' or 'cov', not {cov_terms}")

    # Bin and group the dataframe
    df_bin = _bin_and_group_dataframe_expected_power(
        df_=df_,
        test_cols=test_cols,
        bin_cols_without_df_name=bin_cols_without_df_name,
    )

    # Join the covariance dataframe to df_bin
    df_bin = df_bin.join(df_cov, on=bin_cols_with_df_name, how="left")

    # Set to null any mean values associated with null variance by row/turbine
    df_bin = _synchronize_var_nulls_back_to_mean(df_bin=df_bin, test_cols=test_cols)

    # Get the names of all the mean, covariance and num_points columns
    mean_cols = [f"{col}_mean" for col in test_cols]
    cov_cols = [f"cov_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]
    n_cols = [f"count_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]

    # Synchronize the null values in the mean and all the cov columns across the pairs
    df_bin = _synchronize_nulls(
        df_bin=df_bin,
        sync_cols=mean_cols + cov_cols + n_cols,
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

    # Compute the farm power
    df_bin = df_bin.with_columns(pow_farm=pl.sum_horizontal([f"{c}_mean" for c in test_cols]))

    # Computed the expected farm power variance as the sum of cov_cols / n_cols
    # This is equation 4.16
    df_bin = df_bin.with_columns(
        pow_farm_var=pl.sum_horizontal(
            [pl.col(cov) / pl.col(n) for cov, n in zip(cov_cols, n_cols)]
        )
    )

    # Check if df_bin has any rows
    if df_bin.shape[0] == 0:
        raise ValueError("No rows in df_bin after filtering")

    # Determine the weighting of the ws/wd bins
    df_bin, df_freq_pl = add_bin_weights(df_bin, df_freq_pl, bin_cols_without_df_name, weight_by)

    # Normalize the weight column over the values in df_name column and compute
    # the weighted farm power and weighted var per bin
    df_bin = df_bin.with_columns(weight=pl.col("weight") / pl.col("weight").sum().over("df_name"))

    # Add the weighted farm power column
    df_bin = df_bin.with_columns(weighted_farm_power=pl.col("pow_farm") * pl.col("weight"))

    # Now compute uplifts
    uplift_results = {}
    for uplift_pair, uplift_name in zip(uplift_pairs, uplift_names):
        # Subset the dataframe to the uplift_pair in consideration
        df_sub = df_bin.filter(pl.col("df_name").is_in(uplift_pair))

        # Limit the columns to bin_cols_without_df_name, df_name, weight, pow_farm, pow_farm_var
        df_sub = df_sub.select(
            bin_cols_without_df_name
            + ["df_name", "weight", "count", "weighted_farm_power", "pow_farm", "pow_farm_var"]
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

        # Compute the combined count column
        df_sub = df_sub.with_columns(
            count=(pl.col(f"count_{uplift_pair[0]}") + pl.col(f"count_{uplift_pair[1]}"))
        )

        # Compute the expected power ratio per bin
        df_sub = df_sub.with_columns(
            expected_power_ratio=pl.col(f"pow_farm_{uplift_pair[1]}")
            / pl.col(f"pow_farm_{uplift_pair[0]}")
        )

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

        result_dict["energy_uplift_ctr_pc"] = (result_dict["energy_uplift_ctr"] - 1) * 100
        result_dict["energy_uplift_lb_pc"] = (result_dict["energy_uplift_lb"] - 1) * 100
        result_dict["energy_uplift_ub_pc"] = (result_dict["energy_uplift_ub"] - 1) * 100
        result_dict["count"] = df_sub["count"].sum()

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
    cov_terms: str = "zero",
) -> ExpectedPowerAnalysisOutput:
    """Calculate the total uplift in energy production using expected power methods.

    Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        test_turbines (List[int]): A list of turbine indices to include when computing total
            uplift.
        wd_turbines (List[int]): A list of turbine indices for determining the reference wind
            direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for determining the reference wind
            speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by. Defaults to
            ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin. Defaults to None.
        use_standard_error (bool): If True, uncertainty in the total uplift is quantified by
            propagating bin-level standard errors. If False, bootstrapping is used.
            Defaults to True.
        N (int): The number of bootstrap samples. If use_standard_error is True, N must be 1.
            Defaults to 1.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        cov_terms (str): If use_standard_error is True, the approach for determining the power
            covariance terms betweens pairs of test turbines whe computing uncertainty. Can be
            "zero", "var" or "cov". If "zero" all covariance terms are set to zero. If "var" all
            covariance terms are set to the product of the square root of the variances of each
            individual turbine. If "cov" the computed covariance terms are used as is when enough
            data are present in a bin, with missing terms set to the product of the square root of
            the individual turbine variances. Defaults to "zero".

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
            cov_terms=cov_terms,
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
        cov_terms=cov_terms,
    )

    # Return the object
    return epao


def total_uplift_expected_power_sweep_ws_min(
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
    cov_terms: str = "zero",
    n_step: int = 10,
):
    """Perform a sweep over the ws_min parameter.

    Perform a sweep over the ws_min parameter between the specified ws_min and ws_min + ws_step to
    determine the sensitivity of the total uplift to ws_min. Prints the total uplift for each
    ws_min considered.

    Args:
        a_in (AnalysisInput): An AnalysisInput object containing the dataframes to analyze.
        uplift_pairs (List[Tuple[str, str]]): A list of tuples containing the df_name
            values to compare.
        uplift_names (List[str]): A list of names for the uplift results for each uplift pair.
        test_turbines (List[int]): A list of turbine indices to include when computing total
            uplift.
        wd_turbines (List[int]): A list of turbine indices for determining the reference wind
            direction. Defaults to None.
        ws_turbines (List[int]): A list of turbine indices for determining the reference wind
            speed. Defaults to None.
        use_predefined_wd (bool): Use predefined wind direction. Defaults to False.
        use_predefined_ws (bool): Use predefined wind speed. Defaults to False.
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_in (List[str]): A list of column names to bin the dataframes by. Defaults to
            ["wd_bin", "ws_bin"].
        weight_by (str): The method used to weight the bins when computing total energy if df_freq
            is None ("min" or "sum"). If "min", the minimum number of points in a bin over all
            dataframes is used. If, "sum", the sum of the points over all dataframes is used.
            Defaults to "min".
        df_freq (pd.DataFrame): A pandas dataframe with the frequency of each bin.
            Defaults to None.
        use_standard_error (bool): If True, uncertainty in the total uplift is quantified by
            propagating bin-level standard errors. If False, bootstrapping is used.
            Defaults to True.
        N (int): The number of bootstrap samples. If use_standard_error is True, N must be 1.
            Defaults to 1.
        percentiles (List[float]): The lower and upper percentiles for quantifying uncertainty in
            total uplift. Defaults to [2.5, 97.5].
        remove_any_null_turbine_bins (bool): If True, when computing farm power, remove any bins
            where any of the test turbines is null.  Defaults to False.
        cov_terms (str): If use_standard_error is True, the approach for determining the power
            covariance terms betweens pairs of test turbines whe computing uncertainty. Can be
            "zero", "var" or "cov". If "zero" all covariance terms are set to zero. If "var" all
            covariance terms are set to the product of the square root of the variances of each
            individual turbine. If "cov" the computed covariance terms are used as is when enough
            data are present in a bin, with missing terms set to the product of the square root of
            the individual turbine variances. Defaults to "zero".
        n_step (int): The number of steps to perform the minimum wind speed sweep over.
            Defaults to 10.

    Returns:
        None
    """
    # Get the min_steps to try
    ws_min_values = np.linspace(ws_min, ws_min + ws_step, n_step)

    # Loop over these values
    for ws_min_loop in ws_min_values:
        epao = total_uplift_expected_power(
            a_in=a_in,
            uplift_pairs=uplift_pairs,
            uplift_names=uplift_names,
            test_turbines=test_turbines,
            wd_turbines=wd_turbines,
            ws_turbines=ws_turbines,
            use_predefined_wd=use_predefined_wd,
            use_predefined_ws=use_predefined_ws,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min_loop,
            ws_max=ws_max,
            bin_cols_in=bin_cols_in,
            weight_by=weight_by,
            df_freq=df_freq,
            use_standard_error=use_standard_error,
            N=N,
            percentiles=percentiles,
            remove_any_null_turbine_bins=remove_any_null_turbine_bins,
            cov_terms=cov_terms,
        )

        # Yield the results
        out_string = epao._return_uplift_string()
        print(f"ws_min: {ws_min_loop:.2f} --- {out_string}")
