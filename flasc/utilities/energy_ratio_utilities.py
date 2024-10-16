"""Utility functions for calculating energy ratios."""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np
import polars as pl


# TODO: Someday I think can replace with polars-native code: https://github.com/pola-rs/polars/issues/8551
def cut(
    col_name: str,
    edges: Union[np.ndarray, list],
) -> pl.Expr:
    """Bins the values in the specified column according to the given edges.

    Args:
        col_name (str): The name of the column to bin.
        edges (array-like): The edges of the bins. Values will be placed into the bin
                            whose left edge is the largest edge less than or equal to
                            the value, and whose right edge is the smallest edge
                            greater than the value.

    Returns:
    expression: An expression object that can be used to bin the column.
    """
    c = pl.col(col_name)
    labels = edges[:-1] + np.diff(edges) / 2.0
    expr = pl.when(c < edges[0]).then(None)
    for edge, label in zip(edges[1:], labels):
        expr = expr.when(c < edge).then(label)
    expr = expr.otherwise(None)

    return expr


def bin_column(
    df_: pl.DataFrame,
    col_name: str,
    bin_col_name: str,
    edges: Union[np.ndarray, list],
) -> pl.DataFrame:
    """Bins the values in the specified column of a Polars DataFrame according to the given edges.

    Args:
        df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
        col_name (str): The name of the column to bin.
        bin_col_name (str): The name to give the new column containing the bin labels.
        edges (array-like): The edges of the bins. Values will be placed into the bin
                            whose left edge is the largest edge less than or equal to
                            the value, and whose right edge is the smallest edge
                            greater than the value.

    Returns:
        pl.DataFrame: A new Polars DataFrame with an additional column containing the bin labels.
    """
    return df_.with_columns(
        cut(col_name=col_name, edges=edges).alias(bin_col_name).cast(df_[col_name].dtype)
    )


def add_ws(df_: pl.DataFrame, ws_cols: List[str], remove_all_nulls: bool = False) -> pl.DataFrame:
    """Add the ws column to a dataframe, given which columns to average over.

    Args:
        df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
        ws_cols (list(str)): The name of the columns to average across.
        remove_all_nulls: (bool): Remove all null values in ws_cols (rather than any)

    Returns:
        pl.DataFrame: A new Polars DataFrame with an additional ws column
    """
    df_with_mean_ws = (
        # df_.select(pl.exclude('ws_bin')) # In case ws_bin already exists
        df_.with_columns(
            # df_.select(ws_cols).mean(axis=1).alias('ws_bin')
            ws=pl.concat_list(ws_cols).list.mean()  # Initially ws_bin is just the mean
        )
        .filter(
            pl.all_horizontal(pl.col(ws_cols).is_not_null())
            if remove_all_nulls
            else pl.any_horizontal(pl.col(ws_cols).is_not_null())
        )
        .filter((pl.col("ws").is_not_null()))
    )

    return df_with_mean_ws


def add_ws_bin(
    df_: pl.DataFrame,
    ws_cols: List[str],
    ws_step: float = 1.0,
    ws_min: float = -0.5,
    ws_max: float = 50.0,
    edges: Optional[Union[np.ndarray, list]] = None,
    remove_all_nulls: bool = False,
) -> pl.DataFrame:
    """Add the ws_bin column to a dataframe.

    Given which columns to average over and the step sizes to use

    Args:
        df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
        ws_cols (list(str)): The name of the columns to average across.
        ws_step (float): Step size for binning
        ws_min (float): Minimum wind speed
        ws_max (float): Maximum wind speed
        edges (array-like): The edges of the bins. Values will be placed into the bin
                            whose left edge is the largest edge less than or equal to
                            the value, and whose right edge is the smallest edge
                            greater than the value.  Defaults to None, in which case
                            the edges are generated using ws_step, ws_min, and ws_max.
        remove_all_nulls: (bool): Remove all null values in ws_cols (rather than any)

    Returns:
        pl.DataFrame: A new Polars DataFrame with an additional ws_bin column
    """
    if edges is None:
        edges = np.arange(ws_min, ws_max + ws_step, ws_step)

    # Check if edges is a list or numpy array or similar
    elif len(edges) < 2:
        raise ValueError("edges must have length of at least 2")

    df_with_mean_ws = add_ws(df_, ws_cols, remove_all_nulls)

    # Filter to min and max
    df_with_mean_ws = df_with_mean_ws.filter(
        (pl.col("ws") >= ws_min)  # Filter the mean wind speed
        & (pl.col("ws") < ws_max)
    )

    return bin_column(df_with_mean_ws, "ws", "ws_bin", edges)


def add_wd(df_: pl.DataFrame, wd_cols: List[str], remove_all_nulls: bool = False) -> pl.DataFrame:
    """Add the wd column to a dataframe, given which columns to average over.

    Args:
        df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
        wd_cols (list(str)): The name of the columns to average across.
        remove_all_nulls: (bool): Remove all null values in wd_cols (rather than any)

    Returns:
        pl.DataFrame: A new Polars DataFrame with an additional wd column
    """
    # Gather up intermediate column names and final column names
    wd_cols_cos = [c + "_cos" for c in wd_cols]
    wd_cols_sin = [c + "_sin" for c in wd_cols]
    cols_to_return = df_.columns
    if "wd" not in cols_to_return:
        cols_to_return = cols_to_return + ["wd"]

    df_with_mean_wd = (
        # df_.select(pl.exclude('wd_bin')) # In case wd_bin already exists
        df_.filter(
            pl.all_horizontal(pl.col(wd_cols).is_not_null())
            if remove_all_nulls
            else pl.any_horizontal(pl.col(wd_cols).is_not_null())
        )
        # Add the cosine columns
        .with_columns(
            [
                pl.col(wd_cols).mul(np.pi / 180).cos().name.suffix("_cos"),
                pl.col(wd_cols).mul(np.pi / 180).sin().name.suffix("_sin"),
            ]
        )
    )
    df_with_mean_wd = (
        df_with_mean_wd.with_columns(
            [
                # df_with_mean_wd.select(wd_cols_cos).mean(axis=1).alias('cos_mean'),
                # df_with_mean_wd.select(wd_cols_sin).mean(axis=1).alias('sin_mean'),
                pl.concat_list(wd_cols_cos).list.mean().alias("cos_mean"),
                pl.concat_list(wd_cols_sin).list.mean().alias("sin_mean"),
            ]
        )
        .with_columns(
            wd=np.mod(
                pl.reduce(np.arctan2, [pl.col("sin_mean"), pl.col("cos_mean")]).mul(180 / np.pi),
                360.0,
            )
        )
        .filter((pl.col("wd").is_not_null()))
        .select(cols_to_return)  # Select for just the columns we want to return
    )

    return df_with_mean_wd


# (df_, wd_cols, wd_step=2.0, wd_min=0.0, wd_max=360.0, edges=None):@#
def add_wd_bin(
    df_: pl.DataFrame,
    wd_cols: List[str],
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    edges: Optional[Union[np.ndarray, list]] = None,
    remove_all_nulls: bool = False,
):
    """Add the wd_bin column to a dataframe.

    Given which columns to average over
    and the step sizes to use

    Args:
        df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
        wd_cols (list(str)): The name of the columns to average across.
        wd_step (float): Step size for binning
        wd_min (float): Minimum wind direction
        wd_max (float): Maximum wind direction
        edges (array-like): The edges of the bins. Values will be placed into the bin
                        whose left edge is the largest edge less than or equal to
                        the value, and whose right edge is the smallest edge
                        greater than the value.  Defaults to None, in which case
                        the edges are generated using ws_step, ws_min, and ws_max.
        remove_all_nulls: (bool): Remove all null values in wd_cols (rather than any)

    Returns:
        pl.DataFrame: A new Polars DataFrame with an additional ws_bin column
    """
    if edges is None:
        edges = np.arange(wd_min, wd_max + wd_step, wd_step)

    # If not none, edges must have lenght of at least 2
    elif len(edges) < 2:
        raise ValueError("edges must have length of at least 2")

    # Add in the mean wd column
    df_with_mean_wd = add_wd(df_, wd_cols, remove_all_nulls)

    # Filter to min and max
    df_with_mean_wd = df_with_mean_wd.filter(
        (pl.col("wd") >= wd_min)  # Filter the mean wind speed
        & (pl.col("wd") < wd_max)
    )

    return bin_column(df_with_mean_wd, "wd", "wd_bin", edges)


def add_power_test(
    df_: pl.DataFrame,
    test_cols: List[str],
) -> pl.DataFrame:
    """Add the pow_test column to a dataframe, given which columns to average over.

    Args:
        df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
        test_cols (list(str)): The name of the columns to average across.

    Returns:
        pl.DataFrame: A new Polars DataFrame with an additional pow_test column
    """
    return df_.with_columns(pow_test=pl.concat_list(test_cols).list.mean())


def add_power_ref(df_: pl.DataFrame, ref_cols: List[str]):
    """Add the pow_ref column to a dataframe, given which columns to average over.

    Args:
        df_ (pl.DataFrame): The Polars DataFrame containing the column to bin.
        ref_cols (list(str)): The name of the columns to average across.

    Returns:
        pl.DataFrame: A new Polars DataFrame with an additional pow_ref column
    """
    return df_.with_columns(pow_ref=pl.concat_list(ref_cols).list.mean())


def add_reflected_rows(df_: pl.DataFrame, edges: Union[np.ndarray, list], overlap_distance: float):
    """Add reflected rows to a dataframe.

    Adds rows to a dataframe with where the wind direction is
    reflected around the nearest edge if within overlap_distance

    Given a wind direction DataFrame `df_`, this function adds
    reflected rows to the DataFrame such that each wind direction
    in the original DataFrame has a corresponding reflected wind
    direction. The reflected wind direction is calculated by
    subtracting the wind direction from the nearest edge in `edges`
    and then subtracting that difference again from the
    original wind direction. The resulting wind direction
    is then wrapped around to the range [0, 360) degrees. The function
    returns a new DataFrame with the original rows and the added reflected rows.

    This function enables overlapping bins in the energy ratio functions

    Args:
        df_ : polars.DataFrame
            The DataFrame to add reflected rows to.
        edges : numpy.ndarray
            An array of wind direction edges to use for reflection.
            (Should be same as used in energy ratio)
        overlap_distance : float
            The maximum distance between a wind direction and an edge
            for the wind direction to be considered overlapping.

    Returns:
        polars.DataFrame
            A new DataFrame with the original rows and the added reflected rows.
    """
    df_add = df_.clone()
    wd = df_add["wd"].to_numpy()
    diff_matrix = wd[:, None] - edges
    abs_diff_matrix = np.abs(diff_matrix)
    idx = np.argmin(abs_diff_matrix, axis=1)
    signed_mins = diff_matrix[np.arange(len(diff_matrix)), idx]
    df_add = (
        df_add.with_columns(pl.Series(name="distances", values=signed_mins, dtype=pl.Float32))
        .filter(pl.col("distances").abs() < overlap_distance)
        .with_columns(np.mod((pl.col("wd") - pl.col("distances") * 2), 360.0))
        .drop("distances")
    )

    return pl.concat([df_, df_add])


def filter_all_nulls(
    df_: pl.DataFrame,
    ref_cols: List[str],
    test_cols: List[str],
    ws_cols: List[str],
    wd_cols: List[str],
):
    """Filter dataframe for ALL nulls.

    Filter data by requiring ALL values of ref, test, ws, and wd to be valid
    numbers.

    Args:
        df_ (pl.DataFrame): Polars dataframe possibly containing Null values
        ref_cols (list[str]): A list of columns to use as the reference turbines
        test_cols (list[str]): A list of columns to use as the test turbines
        wd_cols (list[str]): A list of columns to derive the wind directions from
        ws_cols (list[str]): A list of columns to derive the wind speeds from

    Returns:
        pl.DataFrame: A dataframe containing the energy ratio between the two sets of turbines.

    """
    return df_.filter(
        pl.all_horizontal(pl.col(ref_cols + test_cols + ws_cols + wd_cols).is_not_null())
    )


def filter_any_nulls(
    df_: pl.DataFrame,
    ref_cols: List[str],
    test_cols: List[str],
    ws_cols: List[str],
    wd_cols: List[str],
):
    """Filter dataframe for ANY nulls.

    Filter data by requiring ANY of ref, ANY of test, ANY of ws, and ANY of wd
    to be a valid number.

    Args:
        df_ (pl.DataFrame): Polars dataframe possibly containing Null values
        ref_cols (list[str]): A list of columns to use as the reference turbines
        test_cols (list[str]): A list of columns to use as the test turbines
        wd_cols (list[str]): A list of columns to derive the wind directions from
        ws_cols (list[str]): A list of columns to derive the wind speeds from

    Returns:
        pl.DataFrame: A dataframe containing the energy ratio between the two sets of turbines.

    """
    return (
        df_.filter(pl.any_horizontal(pl.col(ref_cols).is_not_null()))
        .filter(pl.any_horizontal(pl.col(test_cols).is_not_null()))
        .filter(pl.any_horizontal(pl.col(ws_cols).is_not_null()))
        .filter(pl.any_horizontal(pl.col(wd_cols).is_not_null()))
    )


def check_compute_energy_ratio_inputs(
    df_,
    ref_turbines,
    test_turbines,
    wd_turbines,
    ws_turbines,
    use_predefined_ref,
    use_predefined_wd,
    use_predefined_ws,
    wd_step,
    wd_min,
    wd_max,
    ws_step,
    ws_min,
    ws_max,
    bin_cols_in,
    weight_by,
    df_freq,
    wd_bin_overlap_radius,
    uplift_pairs,
    uplift_names,
    uplift_absolute,
    N,
    percentiles,
    remove_all_nulls,
):
    """Check the inputs to compute_energy_ratio.

    Check inputs to compute_energy_ratio. Inputs reflect inputs to compute_energy_ratio,
    with exception of df_, which is passed directly instead of er_in.

    All the inputs of compute_energy_ratio are checked for validity. This function does not
    check every input, although they are all accepted.

    Args:
        df_ (pl.DataFrame): The Polars DataFrame
        ref_turbines (list): A list of the reference turbine columns
        test_turbines (list): A list of the test turbine columns
        wd_turbines (list): A list of the wind direction columns
        ws_turbines (list): A list of the wind speed columns
        use_predefined_ref (bool): Whether to use predefined reference turbines
        use_predefined_wd (bool): Whether to use predefined wind direction turbines
        use_predefined_ws (bool): Whether to use predefined wind speed turbines
        wd_step (float): Step size for binning wind direction
        wd_min (float): Minimum wind direction
        wd_max (float): Maximum wind direction
        ws_step (float): Step size for binning wind speed
        ws_min (float): Minimum wind speed
        ws_max (float): Maximum wind speed
        bin_cols_in (list): A list of columns to bin
        weight_by (str): A string indicating how to weight the bins
        df_freq (pl.DataFrame): A DataFrame containing frequency data
        wd_bin_overlap_radius (float): The radius for overlapping wind direction bins
        uplift_pairs (list): A list of uplift pairs
        uplift_names (list): A list of uplift names
        uplift_absolute (bool): Whether to use absolute uplift
        N (int): Number of bootstrapping iterations
        percentiles (list): A list of percentiles to calculate from bootstrap
        remove_all_nulls (bool): Whether to remove all nulls
    """
    # Check that the inputs are valid
    # If use_predefined_ref is True, df_ must have a column named 'pow_ref'
    if use_predefined_ref:
        if "pow_ref" not in df_.columns:
            raise ValueError("df_ must have a column named pow_ref when use_predefined_ref is True")
        # If ref_turbines supplied, warn user that it will be ignored
        if ref_turbines is not None:
            warnings.warn("ref_turbines will be ignored when use_predefined_ref is True")
    else:
        # ref_turbine must be supplied
        if ref_turbines is None:
            raise ValueError("ref_turbines must be supplied when use_predefined_ref is False")

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

    # Confirm that test_turbines is a list of ints or a numpy array of ints
    if not isinstance(test_turbines, list) and not isinstance(test_turbines, np.ndarray):
        raise ValueError("test_turbines must be a list or numpy array of ints")

    # Confirm that test_turbines is not empty
    if len(test_turbines) == 0:
        raise ValueError("test_turbines cannot be empty")

    # Confirm that wd_bin_overlap_radius is less than or equal to wd_step/2
    if wd_bin_overlap_radius > wd_step / 2:
        raise ValueError("wd_bin_overlap_radius must be less than or equal to wd_step/2")

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

    return None


def bin_and_group_dataframe(
    df_: pl.DataFrame,
    ref_cols: List,
    test_cols: List,
    wd_cols: List,
    ws_cols: List,
    wd_step: float = 2.0,
    wd_min: float = 0.0,
    wd_max: float = 360.0,
    ws_step: float = 1.0,
    ws_min: float = 0.0,
    ws_max: float = 50.0,
    wd_bin_overlap_radius: float = 0.0,
    remove_all_nulls: bool = False,
    bin_cols_without_df_name: List = None,
    num_df: int = 0,
):
    """Bin and aggregate a DataFrame based on wind direction and wind speed parameters.

    This function takes a Polars DataFrame (df_) and performs
    binning and aggregation operations based on
    wind direction (wd) and wind speed (ws). It allows for optional
    handling of reflected rows and grouping by
    specific columns. The resulting DataFrame contains aggregated
    statistics for reference and test power
    columns within specified bins.

    Args:
        df_ (DataFrame): The input Polars DataFrame to be processed.
        ref_cols (List[str]): List of columns containing reference power data.
        test_cols (List[str]): List of columns containing test power data.
        wd_cols (List[str]): List of columns containing wind direction data.
        ws_cols (List[str]): List of columns containing wind speed data.
        wd_step (float, optional): Step size for wind direction binning. Defaults to 2.0.
        wd_min (float, optional): Minimum wind direction value. Defaults to 0.0.
        wd_max (float, optional): Maximum wind direction value. Defaults to 360.0.
        ws_step (float, optional): Step size for wind speed binning. Defaults to 1.0.
        ws_min (float, optional): Minimum wind speed value. Defaults to 0.0.
        ws_max (float, optional): Maximum wind speed value. Defaults to 50.0.
        wd_bin_overlap_radius (float, optional): Radius for overlapping wind direction bins.
             Defaults to 0.0.
        remove_all_nulls (bool, optional): If True, remove rows unless all valid instead of any.
            Defaults to False.
        bin_cols_without_df_name (List[str], optional): List of columns used
            for grouping without 'df_name'.
        num_df (int, optional): Number of dataframes required for each bin combination.

    Returns:
        DataFrame: The resulting Polars DataFrame with aggregated statistics.
    """
    # If wd_bin_overlap_radius is not zero, add reflected rows
    if wd_bin_overlap_radius > 0.0:
        # Need to obtain the wd column now rather than during binning
        df_ = add_wd(df_, wd_cols, remove_all_nulls)

        # Add reflected rows
        edges = np.arange(wd_min, wd_max + wd_step, wd_step)
        df_ = add_reflected_rows(df_, edges, wd_bin_overlap_radius)

    # Assign the wd/ws bins
    df_ = add_ws_bin(df_, ws_cols, ws_step, ws_min, ws_max, remove_all_nulls=remove_all_nulls)
    df_ = add_wd_bin(df_, wd_cols, wd_step, wd_min, wd_max, remove_all_nulls=remove_all_nulls)

    # Assign the reference and test power columns
    df_ = add_power_ref(df_, ref_cols)
    df_ = add_power_test(df_, test_cols)

    bin_cols_with_df_name = bin_cols_without_df_name + ["df_name"]

    # Group df_
    df_ = (
        df_.filter(
            pl.all_horizontal(pl.col(bin_cols_with_df_name).is_not_null())
        )  # Select for all bin cols present
        .group_by(bin_cols_with_df_name, maintain_order=True)
        .agg([pl.mean("pow_ref"), pl.mean("pow_test"), pl.count()])
        # Enforce that each ws/wd bin combination has to appear in all dataframes
        .filter(pl.count().over(bin_cols_without_df_name) == num_df)
    )

    return df_


def add_bin_weights(
    df_: pl.DataFrame,
    df_freq_pl: pl.DataFrame = None,
    bin_cols_without_df_name: List = None,
    weight_by: str = "min",
):
    """Add weights to DataFrame bins.

    Add weights to DataFrame bins based on either frequency counts or
    the provided frequency table df_freq_pl.

    This function  assigns weights to DataFrame bins.  If 'df_freq_pl' is provided,
    these weights are used directly.  If 'df_freq_pl' is not provided, the function
    calculates the weights from the input DataFrame 'df_'.
    Weights can be determined as either the minimum ('min') or the sum ('sum') of counts.

    Args:
        df_ (DataFrame): The input Polars DataFrame containing bins and frequency counts.
        df_freq_pl (DataFrame, optional): A Polars DataFrame containing frequency counts for bins.
            If not provided, the function will calculate these counts from 'df_'.
        bin_cols_without_df_name (List, optional): List of columns used for grouping
            bins without 'df_name'.
        weight_by (str, optional): Weight calculation method, either 'min'
            (minimum count) or 'sum' (sum of counts).
            Defaults to 'min'.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing the modified DataFrame 'df_'
            with added weights and the DataFrame
    'df_freq_pl' with the calculated frequency counts.

    Raises:
        RuntimeError: If none of the ws/wd bins in data appear in df_freq.
        UserWarning: If some bins in data are not in df_freq and will receive a weight of 0.

    """
    if df_freq_pl is None:
        # Determine the weights per bin as either the min or sum count
        df_freq_pl = (
            df_.select(bin_cols_without_df_name + ["count"])
            .group_by(bin_cols_without_df_name)
            .agg([pl.min("count") if weight_by == "min" else pl.sum("count")])
            .rename({"count": "weight"})
        )

    df_ = df_.join(df_freq_pl, on=["wd_bin", "ws_bin"], how="left").with_columns(pl.col("weight"))

    # Check if all the values in the weight column are null
    if df_["weight"].is_null().all():
        raise RuntimeError("None of the ws/wd bins in data appear in df_freq")

    # Check if any of the values in the weight column are null
    if df_["weight"].is_null().any():
        warnings.warn("Some bins in data are not in df_freq and will get 0 weight")

    # Fill the null values with zeros
    df_ = df_.with_columns(pl.col("weight").fill_null(strategy="zero"))

    # Normalize the weights
    df_ = df_.with_columns(pl.col("weight").truediv(pl.col("weight").sum()))

    return df_, df_freq_pl
