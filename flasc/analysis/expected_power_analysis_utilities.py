"""Utilities for SCADA data using expected power methods."""

from itertools import product
from typing import List

import polars as pl

from flasc.logging_manager import LoggingManager
from flasc.utilities.energy_ratio_utilities import add_wd_bin, add_ws_bin

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def _add_wd_ws_bins(
    df_: pl.DataFrame,
    wd_cols: List[str],
    ws_cols: List[str],
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    remove_all_nulls_wd_ws: bool = False,
) -> pl.DataFrame:
    """Add wind direction (wd) and wind speed (ws) bin columns to the dataframe.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        wd_cols (List[str]): A list of column names for wind direction
        ws_cols (List[str]): A list of column names for wind speed
        wd_step (float): The step size for the wind direction bins. Defaults to 2.0.
        wd_min (float): The minimum wind direction value. Defaults to 0.0.
        wd_max (float): The maximum wind direction value. Defaults to 360.0.
        ws_step (float): The step size for the wind speed bins. Defaults to 1.0.
        ws_min (float): The minimum wind speed value. Defaults to 0.0.
        ws_max (float): The maximum wind speed value. Defaults to 50.0.
        remove_all_nulls_wd_ws (bool): Remove all null cases for wind direction and wind speed.
            Defaults to False.

    Returns:
        pl.DataFrame: A polars dataframe with the wd and ws bin columns added.
    """
    df_ = add_ws_bin(df_, ws_cols, ws_step, ws_min, ws_max, remove_all_nulls=remove_all_nulls_wd_ws)
    df_ = add_wd_bin(df_, wd_cols, wd_step, wd_min, wd_max, remove_all_nulls=remove_all_nulls_wd_ws)
    return df_


def _bin_and_group_dataframe_expected_power(
    df_: pl.DataFrame,
    test_cols: List[str],
    bin_cols_without_df_name: List[str] = ["wd_bin", "ws_bin"],
) -> pl.DataFrame:
    """Group dataframes by bin columns and calculate the mean and variance of the test columns.

    Args:
        df_ (pl.DataFrame): A polars dataframe, exported from a_in.get_df()
        test_cols (List[str]): A list of column names to calculate the mean and variance of
        bin_cols_without_df_name (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].

    Returns:
        pl.DataFrame: A polars dataframe with the mean and variance of the test columns
            grouped by bin columns.
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
    sync_cols: List[str],
    uplift_pairs: List[List[str]],
    bin_cols_without_df_name: List[str] = ["wd_bin", "ws_bin"],
) -> pl.DataFrame:
    """Copy the nans from the test columns in one of df_name to the other within uplift pairs.

    Args:
        df_bin (pl.DataFrame): A polars dataframe with the mean and variance of the test
            columns grouped by bin columns.
        sync_cols (List[str]): Columns to synchronize.
        uplift_pairs (List[List[str]]): A list of the df_name values to copy the nans from
            for each pair
        bin_cols_without_df_name (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].

    Returns:
        pl.DataFrame: A polars dataframe with the nans copied from one of the df_name values
             to the other.
    """
    df_ = df_bin.clone()

    for uplift_pair in uplift_pairs:
        #   Filter the DataFrame to include only rows where the setting is in uplift_pair
        filtered_df = df_.filter(pl.col("df_name").is_in(uplift_pair))

        # First, create a mask DataFrame indicating where nulls are present for each pow_*_mean
        #    column
        mask_df = filtered_df.select(
            bin_cols_without_df_name
            + ["df_name"]
            + [pl.col(col).is_null().alias(f"{col}_is_null") for col in sync_cols]
        )

        # Group by the bin columns and setting column, and take the maximum of null presence
        # for each setting
        max_nulls = mask_df.group_by(bin_cols_without_df_name).agg(
            [pl.col(f"{col}_is_null").max().alias(f"{col}_should_be_null") for col in sync_cols]
        )

        # Join the mask back to the original DataFrame
        joined_df = df_.join(max_nulls, on=bin_cols_without_df_name, how="left")

        # Set the columns to null only for rows in uplift_pair, according to the
        # synchronized null masks
        updated_columns = [
            pl.when((pl.col("df_name").is_in(uplift_pair)) & (pl.col(f"{col}_should_be_null")))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in sync_cols
        ]

        df_ = joined_df.with_columns(updated_columns).select(df_.columns)  # Return original columns

    return df_


def _get_num_points(
    df_: pl.DataFrame,
    test_cols: List[str],
    bin_cols_with_df_name: List[str],
) -> pl.DataFrame:
    """Get the number of points for each pair of test columns.

    Args:
        df_ (pl.DataFrame): A polars dataframe
        test_cols (List[str]): A list of column names to calculate the number of points for
        bin_cols_with_df_name (List[str]): A list of column names to bin the dataframes by.

    Returns:
        pl.DataFrame: A polars dataframe with the number of points for each pair of test columns.
    """
    # Generate all pairs of columns (including same column pairs)
    col_pairs = list(product(test_cols, test_cols))

    df_n = df_.group_by(bin_cols_with_df_name).agg(pl.count().alias("count"))

    for c1, c2 in col_pairs:
        df_sub = df_.filter(
            pl.col(c1).is_not_null()
            & pl.col(c2).is_not_null()
            & pl.col(c1).is_not_nan()
            & pl.col(c2).is_not_nan()
        )
        df_n_sub = df_sub.group_by(bin_cols_with_df_name).agg(pl.count().alias(f"count_{c1}_{c2}"))

        df_n = df_n.join(df_n_sub, on=bin_cols_with_df_name, how="left")

    # Fill nulls to 0
    df_n = df_n.fill_null(0)

    # Sort by bin_cols_with_df_name
    df_n = df_n.sort(bin_cols_with_df_name)

    return df_n


def _compute_covariance(
    df_: pl.DataFrame,
    test_cols: List[str],
    bin_cols_with_df_name: List[str],
) -> pl.DataFrame:
    """Compute the covariance matrix for the test columns.

    Args:
        df_ (pl.DataFrame): A polars dataframe
        test_cols (List[str]): A list of column names to calculate the covariance of
        bin_cols_with_df_name (List[str]): A list of column names to bin the dataframes by.

    Returns:
        pl.DataFrame: A polars dataframe with the covariance matrix.
    """
    # Generate all pairs of columns (including same column pairs)
    col_pairs = list(product(test_cols, test_cols))

    # Create expressions for all covariances
    cov_exprs = [pl.cov(pair[0], pair[1]).alias(f"cov_{pair[0]}_{pair[1]}") for pair in col_pairs]

    # Compute covariances
    df_cov = df_.group_by(bin_cols_with_df_name).agg(cov_exprs)

    # Get the number of points for each pair
    df_n = _get_num_points(df_, test_cols=test_cols, bin_cols_with_df_name=bin_cols_with_df_name)

    # Join the number of points to the covariance matrix
    df_cov = df_cov.join(df_n, on=bin_cols_with_df_name, how="left")

    return df_cov


def _null_and_sync_covariance(
    df_cov: pl.DataFrame,
    test_cols: List[str],
    uplift_pairs: List[List[str]],
    bin_cols_without_df_name: List[str] = ["wd_bin", "ws_bin"],
) -> pl.DataFrame:
    """Apply null values to the covariance matrix and synchronize across uplift pairs.

    Args:
        df_cov (pl.DataFrame): A polars dataframe with the covariance matrix
        test_cols (List[str]): A list of column names to calculate the covariance of
        uplift_pairs (List[List[str]]): A list of the df_name values to copy the nans from
        bin_cols_without_df_name (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].

    Returns:
        pl.DataFrame: A polars dataframe with the null values applied and synchronized across
            uplift pairs.
    """
    # Get the names of all the covariance and num_points columns
    cov_cols = [f"cov_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]
    n_cols = [f"count_{t1}_{t2}" for t1, t2 in product(test_cols, test_cols)]

    # Convert all NaN to None
    df_cov = df_cov.fill_nan(None)

    # Loop over cov_cols and n_cols
    for cov_col, n_col in zip(cov_cols, n_cols):
        # Wherever n_col 0, 1  set the corresponding cov_col to NaN
        df_cov = df_cov.with_columns(
            pl.when(pl.col(n_col) < 2)  # | pl.col(n_col).is_null())
            .then(None)
            .otherwise(pl.col(cov_col))
            .alias(cov_col)
        )

        # Wherever n_col is null, set cov_col to null
        df_cov = df_cov.with_columns(
            pl.when(pl.col(n_col).is_null()).then(None).otherwise(pl.col(cov_col)).alias(cov_col)
        )

        # Wherever n_col is 0, set to null
        df_cov = df_cov.with_columns(
            pl.when(pl.col(n_col) == 0).then(None).otherwise(pl.col(n_col)).alias(n_col)
        )

    # Now sync across df_names
    sync_cols = cov_cols + n_cols

    df_cov = _synchronize_nulls(
        df_bin=df_cov,
        sync_cols=sync_cols,
        uplift_pairs=uplift_pairs,
        bin_cols_without_df_name=bin_cols_without_df_name,
    )

    return df_cov


def _fill_cov_null(
    df_cov: pl.DataFrame,
    test_cols: List[str],
    uplift_pairs: List[List[str]],
    bin_cols_without_df_name: List[str] = ["wd_bin", "ws_bin"],
    fill_null_cov_bin_strategy="max_var",
) -> pl.DataFrame:
    """Fill null values in covariance according to strategy.

    Fill the null values in the covariance matrix according to the
    strategy specified in fill_null_cov_bin_strategy.
    If max_var, fill null values of covariance of t1,t2 with the
    combined variances of t1 and t2 assuming correlation
    of 1.0.  For min var...

    Args:
        df_cov (pl.DataFrame): A polars dataframe with the covariance matrix
        test_cols (List[str]): A list of column names to calculate the covariance of
        uplift_pairs (List[List[str]]): A list of the df_name values to copy the nans from
        bin_cols_without_df_name (List[str]): A list of column names to bin the dataframes by.
            Defaults to ["wd_bin", "ws_bin"].
        fill_null_cov_bin_strategy (str): Strategy to fill null values in covariance matrix.
            Defaults to 'max_var'.

    Returns:
        pl.DataFrame: A polars dataframe with the null values filled according to the strategy.
    """
    pass