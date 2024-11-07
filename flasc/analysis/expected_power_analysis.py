"""Analyze SCADA data using expected power methods."""

import warnings

import numpy as np
import pandas as pd
import polars as pl

from flasc.analysis.analysis_input import AnalysisInput
from flasc.logging_manager import LoggingManager
from flasc.utilities.energy_ratio_utilities import add_wd_bin, add_ws_bin

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def _bin_and_group_dataframe_expected_power(
    df_: pl.DataFrame,
    test_cols: list,
    wd_cols: list,
    ws_cols: list,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    bin_cols_without_df_name: list = ["wd_bin", "ws_bin"],
    remove_all_nulls_wd_ws: bool = False,
) -> pl.DataFrame:
    """Group dataframes by bin_cols_without_df_name.

    Group dataframes by bin_cols_without_df_name and calculate the mean and variance of the test_cols.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        test_cols (list): A list of column names to calculate the mean and variance of
        wd_cols (list): A list of column names for wind direction
        ws_cols (list): A list of column names for wind speed
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        bin_cols_without_df_name (list): A list of column names to bin the dataframes by. Defaults to ["wd_bin", "ws_bin"].
        remove_all_nulls_wd_ws (bool): Remove all null cases for wind direction and wind speed. Defaults to False.

    Returns:
        pl.DataFrame: A polars dataframe with the mean and variance of the test_cols grouped by bin_cols_without_df_name
    """
    num_df = df_["df_name"].n_unique()

    # Remove all null cases
    # Assign the wd/ws bins
    df_ = add_ws_bin(df_, ws_cols, ws_step, ws_min, ws_max, remove_all_nulls=remove_all_nulls_wd_ws)
    df_ = add_wd_bin(df_, wd_cols, wd_step, wd_min, wd_max, remove_all_nulls=remove_all_nulls_wd_ws)

    bin_cols_with_df_name = bin_cols_without_df_name + ["df_name"]

    # Group df_
    df_ = (
        df_.filter(
            pl.all_horizontal(pl.col(bin_cols_with_df_name).is_not_null())
        )  # Select for all bin cols present
        .group_by(bin_cols_with_df_name, maintain_order=True)
        # # Compute the mean and standard deviation of each of test_cols
        .agg(
            [pl.mean(c).alias(f"{c}_mean") for c in test_cols]
            + [pl.var(c).alias(f"{c}_var") for c in test_cols]
            + [pl.count(c).alias(f"{c}_count") for c in test_cols]
        )
        # Drop any row in which all the mean values are null
        .filter(pl.any_horizontal([pl.col(f"{c}_mean").is_not_null() for c in test_cols]))
        # # Enforce that each ws/wd bin combination has to appear in all dataframes
        .filter(pl.count().over(bin_cols_without_df_name) == num_df)
    )

    return df_


def _total_uplift_expected_power_single(
    a_in: AnalysisInput,
    test_cols: list,
    wd_cols: list,
    ws_cols: list,
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
    remove_all_nulls_wd_ws: bool = False,
):
    # Get the polars dataframe from within the a_in
    df_ = a_in.get_df()

    # Get the dataframe names
    df_names = a_in.df_names

    # Get the number of dataframes
    num_df = len(df_names)

    bin_cols_without_df_name = [c for c in bin_cols_in if c != "df_name"]


def _total_uplift_expected_power_with_standard_error():
    pass


def _total_uplift_expected_power_with_bootstrapping():
    pass


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
