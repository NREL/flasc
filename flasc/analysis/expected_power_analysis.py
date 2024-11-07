"""Analyze SCADA data using expected power methods."""

import warnings

import numpy as np
import pandas as pd
import polars as pl

from flasc.analysis.analysis_input import AnalysisInput
from flasc.logging_manager import LoggingManager
from flasc.utilities.energy_ratio_utilities import add_bin_weights, add_wd_bin, add_ws_bin

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def _add_wd_ws_bins(
    df_: pl.DataFrame,
    wd_cols: list,
    ws_cols: list,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    remove_all_nulls_wd_ws: bool = False,
):
    """Add wd and ws bin columns.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        wd_cols (list): A list of column names for wind direction
        ws_cols (list): A list of column names for wind speed
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        remove_all_nulls_wd_ws (bool): Remove all null cases for wind direction and wind speed. Defaults to False.

    Returns:
        pl.DataFrame: A polars dataframe with the wd and ws bin columns added
    """
    df_ = add_ws_bin(df_, ws_cols, ws_step, ws_min, ws_max, remove_all_nulls=remove_all_nulls_wd_ws)
    df_ = add_wd_bin(df_, wd_cols, wd_step, wd_min, wd_max, remove_all_nulls=remove_all_nulls_wd_ws)
    return df_


def _bin_and_group_dataframe_expected_power(
    df_: pl.DataFrame,
    test_cols: list,
    bin_cols_without_df_name: list = ["wd_bin", "ws_bin"],
) -> pl.DataFrame:
    """Group dataframes by bin_cols_without_df_name.

    Group dataframes by bin_cols_without_df_name and calculate the mean and variance of the test_cols.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        test_cols (list): A list of column names to calculate the mean and variance of
        bin_cols_without_df_name (list): A list of column names to bin the dataframes by. Defaults to ["wd_bin", "ws_bin"].
        

    Returns:
        pl.DataFrame: A polars dataframe with the mean and variance of the test_cols grouped by bin_cols_without_df_name
    """
    num_df = df_["df_name"].n_unique()

    bin_cols_with_df_name = bin_cols_without_df_name + ["df_name"]

    # Group df_
    df_ = (
        df_.filter(pl.all_horizontal(pl.col(bin_cols_with_df_name).is_not_null()))
        # Drop rows where all test_cols are null
        .filter(pl.any_horizontal([pl.col(c).is_not_null() for c in test_cols]))
        # Select for all bin cols present
        .group_by(bin_cols_with_df_name, maintain_order=True)
        # # Compute the mean and standard deviation of each of test_cols
        .agg(
            [pl.mean(c).alias(f"{c}_mean") for c in test_cols]
            + [pl.var(c).alias(f"{c}_var") for c in test_cols]
            + [pl.count(c).alias(f"{c}_count") for c in test_cols]
            + [pl.count().alias("count")]
        )
        # Drop any row in which all the mean values are null
        .filter(pl.any_horizontal([pl.col(f"{c}_mean").is_not_null() for c in test_cols]))
        # # Enforce that each ws/wd bin combination has to appear in all dataframes
        .filter(pl.count().over(bin_cols_without_df_name) == num_df)
    )

    return df_


def _synchronize_nulls(
    df_bin: pl.DataFrame,
    sync_cols: list,
    uplift_pairs: list,
    bin_cols_without_df_name: list = ["wd_bin", "ws_bin"],
) -> pl.DataFrame:
    """Copy the nans from the test columns in one of the df_name values to the other.

    Args:
        df_bin (pl.DataFrame): A polars dataframe with the mean and variance of the test_cols
            grouped by bin_cols_with_df_name
        sync_cols (list): Cols to synchronize
        uplift_pairs (list): A list of the df_name values to copy the nans from
        bin_cols_without_df_name (list): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].

    Returns:
        pl.DataFrame: A polars dataframe with the nans copied from one of the df_name
            values to the other
    """
    df_ = df_bin.clone()

    for uplift_pair in uplift_pairs:
        #   Filter the DataFrame to include only rows where the setting is in uplift_pair
        filtered_df = df_.filter(pl.col("df_name").is_in(uplift_pair))

        # First, create a mask DataFrame indicating where nulls are present for each pow_*_mean column
        mask_df = filtered_df.select(
            bin_cols_without_df_name
            + ["df_name"]
            + [pl.col(col).is_null().alias(f"{col}_is_null") for col in sync_cols]
        )

        # Group by the bin columns and setting column, and take the maximum of null presence for each setting
        max_nulls = mask_df.group_by(bin_cols_without_df_name).agg(
            [pl.col(f"{col}_is_null").max().alias(f"{col}_should_be_null") for col in sync_cols]
        )

        # Join the mask back to the original DataFrame
        joined_df = df_.join(max_nulls, on=bin_cols_without_df_name, how="left")

        # Set the columns to null only for rows in uplift_pair, according to the synchronized null masks
        updated_columns = [
            pl.when((pl.col("df_name").is_in(uplift_pair)) & (pl.col(f"{col}_should_be_null")))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in sync_cols
        ]

        df_ = joined_df.with_columns(updated_columns).select(df_.columns)  # Return original columns

    return df_


def _total_uplift_expected_power_single(
    df_: pl.DataFrame,
    test_cols: list,
    wd_cols: list,
    ws_cols: list,
    uplift_pairs: list,
    uplift_names: list,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: list = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq_pl: pl.DataFrame = None,
    remove_all_nulls_wd_ws: bool = False,
    remove_any_null_turbine_bins: bool = False,
):
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
        remove_all_nulls_wd_ws=remove_all_nulls_wd_ws,
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

    # Remove rows where any of the test columns are null
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

    # with pl.Config(tbl_cols=-1):
    #     print(df_bin)
    #     print(df_sum)
    #     print(uplift_results)

    return df_bin, df_sum, uplift_results


def _total_uplift_expected_power_with_bootstrapping(
    a_in: AnalysisInput,
    test_cols: list,
    wd_cols: list,
    ws_cols: list,
    uplift_pairs: list,
    uplift_names: list,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: list = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq_pl: pl.DataFrame = None,
    remove_all_nulls_wd_ws: bool = False,
    remove_any_null_turbine_bins: bool = False,
    N: int = 1,
    percentiles: list = [2.5, 97.5],
):
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
            remove_all_nulls_wd_ws=remove_all_nulls_wd_ws,
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

    return bootstrap_uplift_result


def _compute_covariance(df_: pl.DataFrame, test_cols: list, bin_cols_with_df_name: list):
    return df_.group_by(bin_cols_with_df_name).agg(pl.cov(test_cols[0], test_cols[1]))


def _total_uplift_expected_power_with_standard_error(
    df_: pl.DataFrame,
    test_cols: list,
    wd_cols: list,
    ws_cols: list,
    uplift_pairs: list,
    uplift_names: list,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: list = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq_pl: pl.DataFrame = None,
    remove_all_nulls_wd_ws: bool = False,
    remove_any_null_turbine_bins: bool = False,
):
    pass

    # Get the covariance matrix between turbines

    # with pl.Config(tbl_cols=-1):
    #     print(df_bin)
    #     print(df_sum)
    #     print(uplift_results)

    # # Get the bin cols without df_name
    # bin_cols_without_df_name = [c for c in bin_cols_in if c != "df_name"]

    # # Add the wd and ws bins
    # df_ = _add_wd_ws_bins(
    #     df_,
    #     wd_cols=wd_cols,
    #     ws_cols=ws_cols,
    #     wd_step=wd_step,
    #     wd_min=wd_min,
    #     wd_max=wd_max,
    #     ws_step=ws_step,
    #     ws_min=ws_min,
    #     ws_max=ws_max,
    #     remove_all_nulls_wd_ws=remove_all_nulls_wd_ws,
    # )

    # # Bin and group the dataframe
    # df_bin = _bin_and_group_dataframe_expected_power(
    #     df_=df_,
    #     test_cols=test_cols,
    #     wd_cols=wd_cols,
    #     ws_cols=ws_cols,
    #     wd_step=wd_step,
    #     wd_min=wd_min,
    #     wd_max=wd_max,
    #     ws_step=ws_step,
    #     ws_min=ws_min,
    #     ws_max=ws_max,
    #     bin_cols_without_df_name=bin_cols_without_df_name,
    #     remove_all_nulls_wd_ws=remove_all_nulls_wd_ws,
    # )

    # # Synchronize the null values
    # df_bin = _synchronize_nulls(
    #     df_bin=df_bin,
    #     sync_cols=[f"{col}_mean" for col in test_cols],
    #     uplift_pairs=uplift_pairs,
    #     bin_cols_without_df_name=bin_cols_without_df_name,
    # )

    # # Remove rows where any of the test columns are null
    # if remove_any_null_turbine_bins:
    #     df_bin = df_bin.filter(pl.all_horizontal([pl.col(f"{c}_mean").is_not_null() for c in test_cols]))


def total_uplift_expected_power(
    a_in: AnalysisInput,
    test_turbines: list = None,
    wd_turbines: list = None,
    ws_turbines: list = None,
    use_predefined_wd: bool = False,
    use_predefined_ws: bool = False,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_in: list = ["wd_bin", "ws_bin"],
    weight_by: str = "min",  # min or sum
    df_freq: pd.DataFrame = None,
    uplift_pairs: list = None,
    uplift_names: list = None,
    use_standard_error: bool = True,
    N: int = 1,
    percentiles: list = [2.5, 97.5],
    remove_all_nulls_wd_ws: bool = False,
):
    """Calculate the total uplift in energy production using expected power methods."""
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
        epao = _total_uplift_expected_power_single(
            a_in,
            test_cols=test_cols,
            wd_cols=wd_cols,
            ws_cols=ws_cols,
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
            remove_all_nulls_wd_ws=remove_all_nulls_wd_ws,
        )
