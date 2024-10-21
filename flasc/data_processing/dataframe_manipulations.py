"""Module containing methods for FLASC dataframe manipulations."""

from __future__ import annotations

import datetime
import os as os
import warnings
from typing import List, Optional, Union
from xmlrpc.client import Boolean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floris.utilities import wrap_360

from flasc import FlascDataFrame
from flasc.logging_manager import LoggingManager
from flasc.utilities import floris_tools as ftools, utilities as fsut

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


# Functions related to wind farm analysis for df
def filter_df_by_ws(
    df: Union[pd.DataFrame, FlascDataFrame], ws_range: List[float]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Filter a dataframe by wind speed range.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements.
        ws_range ([float, float]): Wind speed range [lower bound, upper bound].

    Returns:
        pd.DataFrame | FlascDataFrame: Filtered dataframe.
    """
    df = df[df["ws"] >= ws_range[0]]
    df = df[df["ws"] < ws_range[1]]
    return df


def filter_df_by_wd(
    df: Union[pd.DataFrame, FlascDataFrame], wd_range: List[float]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Filter a dataframe by wind direction range.

    Args:
        df (pd.DataFrame | FlascDataframe): Dataframe with measurements.
        wd_range ([float, float]): Wind direction range [lower bound, upper bound].

    Returns:
        pd.DataFrame | FlascDataFrame: Filtered dataframe.
    """
    lb = wd_range[0]
    ub = wd_range[1]

    lb = wrap_360(lb)
    if ub > 360.0:
        ub = wrap_360(ub)

    wd_array = wrap_360(df["wd"])
    if lb > ub:
        df = df[((wd_array >= lb) | (wd_array < ub))]
    else:
        df = df[((wd_array >= lb) & (df["wd"] < ub))]
    return df


def filter_df_by_ti(
    df: Union[pd.DataFrame, FlascDataFrame], ti_range: List[float]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Filter a dataframe by turbulence intensity range.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements.
        ti_range ([float, float]): Turbulence intensity range [lower bound, upper bound].

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df = df[df["ti"] >= ti_range[0]]
    df = df[df["ti"] < ti_range[1]]
    return df


def get_num_turbines(df: Union[pd.DataFrame, FlascDataFrame]) -> int:
    """Get the number of turbines in a dataframe.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with turbine data

    Returns:
         int: Number of turbines in the dataframe
    """
    return fsut.get_num_turbines(df)


# Generic functions for column operations
def get_column_mean(
    df: Union[pd.DataFrame, FlascDataFrame],
    col_prefix: str = "pow",
    turbine_list: Optional[Union[List[int], np.ndarray]] = None,
    circular_mean: bool = False,
) -> np.ndarray:
    """Get the mean of a column for a list of turbines.

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements.
        col_prefix (str, optional): Column prefix to use. Defaults to "pow".
        turbine_list ([list, array], optional): List of turbine numbers to use.
            If None, all turbines are used.  Defaults to None.
        circular_mean (bool, optional): Use circular mean. Defaults to False.

    Returns:
        np.array: Mean of the column for the specified turbines.
    """
    if turbine_list is None:
        turbine_list = range(get_num_turbines(df))  # Assume all turbines
    elif isinstance(turbine_list, (int, np.integer)):
        turbine_list = [turbine_list]

    col_names = [col_prefix + "_%03d" % ti for ti in turbine_list]
    array = df[col_names].astype(float)

    if circular_mean:
        # Use unit vectors to calculate the mean
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dir_x = np.nanmean(np.cos(array * np.pi / 180.0), axis=1)
        dir_y = np.nanmean(np.sin(array * np.pi / 180.0), axis=1)

        mean_dirs = np.arctan2(dir_y, dir_x)
        mean_out = wrap_360(mean_dirs * 180.0 / np.pi)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_out = np.nanmean(array, axis=1)

    return mean_out


def _set_col_by_turbines(col_out, col_prefix, df, turbine_numbers, circular_mean):
    if isinstance(turbine_numbers, str):
        if turbine_numbers.lower() == "all":
            turbine_numbers = range(get_num_turbines(df=df))

    df[col_out] = get_column_mean(
        df=df, col_prefix=col_prefix, turbine_list=turbine_numbers, circular_mean=circular_mean
    )
    return df


def _set_col_by_n_closest_upstream_turbines(
    col_out,
    col_prefix,
    df,
    N,
    df_upstream,
    circular_mean,
    turb_no,
    x_turbs,
    y_turbs,
    exclude_turbs=[],
):
    # Can get df_upstream using floris_tools.get_upstream_turbs_floris()
    df.loc[df.index, col_out] = np.nan
    for i in range(df_upstream.shape[0]):
        wd_min = df_upstream.loc[i, "wd_min"]
        wd_max = df_upstream.loc[i, "wd_max"]
        upstr_turbs = df_upstream.loc[i, "turbines"]

        # Calculate distances and get closest N upstream turbines
        upstr_turbs = [ti for ti in upstr_turbs if ti not in exclude_turbs]
        x0 = x_turbs[turb_no]
        y0 = y_turbs[turb_no]
        x_upstr = np.array(x_turbs, dtype=float)[upstr_turbs]
        y_upstr = np.array(y_turbs, dtype=float)[upstr_turbs]
        ds = np.sqrt((x_upstr - x0) ** 2.0 + (y_upstr - y0) ** 2.0)
        upstr_turbs_sorted = np.array(upstr_turbs, dtype=int)[np.argsort(ds)]
        upstr_turbs_n_closest = upstr_turbs_sorted[0:N]

        if wd_min > wd_max:  # Wrap around
            ids = (df["wd"] > wd_min) | (df["wd"] <= wd_max)
        else:
            ids = (df["wd"] >= wd_min) & (df["wd"] < wd_max)

        col_mean = get_column_mean(
            df.loc[ids, :],
            col_prefix=col_prefix,
            turbine_list=upstr_turbs_n_closest,
            circular_mean=circular_mean,
        )
        df.loc[ids, col_out] = col_mean

    return df


def _set_col_by_upstream_turbines(
    col_out, col_prefix, df, df_upstream, circular_mean, exclude_turbs=[]
):
    # Can get df_upstream using floris_tools.get_upstream_turbs_floris()
    df.loc[df.index, col_out] = np.nan
    for i in range(df_upstream.shape[0]):
        wd_min = df_upstream.loc[i, "wd_min"]
        wd_max = df_upstream.loc[i, "wd_max"]
        upstr_turbs = df_upstream.loc[i, "turbines"]

        # Exclude particular turbines
        upstr_turbs = [ti for ti in upstr_turbs if ti not in exclude_turbs]

        if wd_min > wd_max:  # Wrap around
            ids = (df["wd"] > wd_min) | (df["wd"] <= wd_max)
        else:
            ids = (df["wd"] >= wd_min) & (df["wd"] < wd_max)

        col_mean = get_column_mean(
            df.loc[ids, :],
            col_prefix=col_prefix,
            turbine_list=upstr_turbs,
            circular_mean=circular_mean,
        )
        df.loc[ids, col_out] = col_mean

    return df


def _set_col_by_radius_from_turbine(
    col_out,
    col_prefix,
    df,
    turb_no,
    x_turbs,
    y_turbs,
    max_radius,
    circular_mean,
    include_itself=True,
):
    turbs_within_radius = ftools.get_turbs_in_radius(
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        turb_no=turb_no,
        max_radius=max_radius,
        include_itself=include_itself,
    )

    if len(turbs_within_radius) < 1:
        logger.warn("No turbines within proximity. Try to increase radius.")
        return None

    return _set_col_by_turbines(
        col_out=col_out,
        col_prefix=col_prefix,
        df=df,
        turbine_numbers=turbs_within_radius,
        circular_mean=circular_mean,
    )


def _set_col_by_upstream_turbines_in_radius(
    col_out,
    col_prefix,
    df,
    df_upstream,
    turb_no,
    x_turbs,
    y_turbs,
    max_radius,
    circular_mean,
    include_itself=True,
):
    """Add a column of averaged upstream turbine values.

    Add a column called [col_out] to your dataframe, which is the
    mean of the columns pow_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        col_out (str): Column name to be added to the dataframe.
        col_prefix (str): Column prefix to use.
        df (pd.DataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.DataFrame): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
            turb_no (int): Turbine number from which the radius should be
            calculated.
        turb_no (int): Turbine number from which the radius should be
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius (float): Maximum radius for the upstream turbines
            until which they are still considered as relevant/used for the
            calculation of the averaged column quantity.
        circular_mean (bool): Use circular mean.  Defaults to False.
        include_itself (bool, optional): Include the measurements of turbine
            turb_no in the determination of the averaged column quantity. Defaults
            to False.

    Returns:
        df (pd.Dataframe): Dataframe which equals the inserted dataframe
        plus the additional column called [col_ref].
    """
    turbs_in_radius = ftools.get_turbs_in_radius(
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        turb_no=turb_no,
        max_radius=max_radius,
        include_itself=include_itself,
    )

    if len(turbs_in_radius) < 1:
        logger.info("No turbines within proximity. Try to increase radius.")
        return None

    turbs = range(len(x_turbs))
    turbs_outside_radius = [i for i in turbs if i not in turbs_in_radius]
    return _set_col_by_upstream_turbines(
        col_out=col_out,
        col_prefix=col_prefix,
        df=df,
        df_upstream=df_upstream,
        circular_mean=circular_mean,
        exclude_turbs=turbs_outside_radius,
    )


# Helper functions
def set_wd_by_turbines(
    df: Union[pd.DataFrame, FlascDataFrame], turbine_numbers: List[int]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add WD column by list of turbines.

    Add a column called 'wd' in your dataframe with value equal
    to the circular-averaged wind direction measurements of all
    the turbines in turbine_numbers.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
            should be used to calculate the column average.

    Returns:
        df (pd.DataFrame | FlascDataFrame): Dataframe which equals the inserted dataframe
            plus the additional column called 'wd'.
    """
    return _set_col_by_turbines("wd", "wd", df, turbine_numbers, True)


def set_wd_by_all_turbines(
    df: Union[pd.DataFrame, FlascDataFrame],
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add a wind direction column using all turbines.

    Add a column called 'wd' in your dataframe with value equal
    to the circular-averaged wind direction measurements of all
    turbines.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'wd'.
    """
    return _set_col_by_turbines("wd", "wd", df, "all", True)


def set_wd_by_radius_from_turbine(
    df: Union[pd.DataFrame, FlascDataFrame],
    turb_no: int,
    x_turbs: List[float],
    y_turbs: List[float],
    max_radius: float,
    include_itself: bool = True,
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add wind direction column by turbines in radius.

    Add a column called 'wd' to your dataframe, which is the
    mean of the columns wd_%03d for turbines that are within radius
    [max_radius] of the turbine of interest [turb_no].

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        turb_no (int): Turbine number from which the radius should be calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius (float): Maximum radius for the upstream turbines
            until which they are still considered as relevant/used for the
            calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
            turb_no in the determination of the averaged column quantity. Defaults
            to False.

    Returns:
        pd.DataFrame | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'wd'.
    """
    return _set_col_by_radius_from_turbine(
        col_out="wd",
        col_prefix="wd",
        df=df,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=True,
        include_itself=include_itself,
    )


def set_ws_by_turbines(
    df: Union[pd.DataFrame, FlascDataFrame], turbine_numbers: List[int]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add ws column by list of turbines.

    Add a column called 'ws' in your dataframe with value equal
    to the mean wind speed measurements of all the turbines in
    turbine_numbers.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
        typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
        potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
        should be used to calculate the column average.

    Returns:
        pd.DataFrame | FlascDataFrame: Dataframe which equals the inserted dataframe
        plus the additional column called 'ws'.
    """
    return _set_col_by_turbines("ws", "ws", df, turbine_numbers, False)


def set_ws_by_all_turbines(
    df: Union[pd.DataFrame, FlascDataFrame],
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add ws column by all turbines.

    Add a column called 'ws' in your dataframe with value equal
    to the circular-averaged wind direction measurements of all
    turbines.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
            should be used to calculate the column average.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'ws'.
    """
    return _set_col_by_turbines("ws", "ws", df, "all", False)


def set_ws_by_upstream_turbines(
    df: Union[pd.DataFrame, FlascDataFrame], df_upstream, exclude_turbs=[]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add wind speed column using upstream turbines.

    Add a column called 'ws' in your dataframe with value equal
    to the averaged wind speed measurements of all the turbines
    upstream, excluding the turbines listed in exclude_turbs.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.DataFrame): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
            exclude_turbs ([list, array]): array-like variable containing
            turbine indices that should be excluded in determining the column
            mean quantity.
        exclude_turbs ([list, array]): array-like variable containing
            turbine indices that should be excluded in determining the column
            mean quantity.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'ws'.
    """
    return _set_col_by_upstream_turbines(
        col_out="ws",
        col_prefix="ws",
        df=df,
        df_upstream=df_upstream,
        circular_mean=False,
        exclude_turbs=exclude_turbs,
    )


def set_ws_by_upstream_turbines_in_radius(
    df: Union[pd.DataFrame, FlascDataFrame],
    df_upstream: pd.DataFrame,
    turb_no: int,
    x_turbs: List[float],
    y_turbs: List[float],
    max_radius: float,
    include_itself: bool = True,
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add wind speed column using in-radius upstream turbines.

    Add a column called 'ws' to your dataframe, which is the
    mean of the columns pow_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.DataFrame): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
            turb_no (int): Turbine number from which the radius should be
            calculated.
        turb_no (int): Turbine number from which the radius should be
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius (float): Maximum radius for the upstream turbines
            until which they are still considered as relevant/used for the
            calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
            turb_no in the determination of the averaged column quantity. Defaults
            to False.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
        plus the additional column called 'ws'.
    """
    return _set_col_by_upstream_turbines_in_radius(
        col_out="ws",
        col_prefix="ws",
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=False,
        include_itself=include_itself,
    )


def set_ws_by_n_closest_upstream_turbines(
    df: Union[pd.DataFrame, FlascDataFrame],
    df_upstream: pd.DataFrame,
    turb_no: int,
    x_turbs: List[float],
    y_turbs: List[float],
    exclude_turbs: List[int] = [],
    N: int = 5,
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add wind speed column by N closest upstream turbines.

    Add a column called 'ws' to your dataframe, which is the
    mean of the columns ws_%03d for the N closest turbines that are
    upstream of the turbine of interest [turb_no].

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.DataFrame): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
            turb_no (int): Turbine number from which the radius should be
            calculated.
        turb_no (int): Turbine number from which the radius should be
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        exclude_turbs ([list, array]): array-like variable containing
            turbine indices that should be excluded in determining the column
            mean quantity.
        N (int): Number of closest turbines to consider for the calculation
    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
        plus the additional column called 'pow_ref'.
    """
    return _set_col_by_n_closest_upstream_turbines(
        col_out="ws",
        col_prefix="ws",
        N=N,
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        exclude_turbs=exclude_turbs,
        circular_mean=False,
    )


def set_ti_by_turbines(
    df: Union[pd.DataFrame, FlascDataFrame], turbine_numbers: List[int]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add TI column by list of turbines.

    Add a column called 'ti' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all the
    turbines listed in turbine_numbers.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
            should be used to calculate the column average.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'ti'.
    """
    return _set_col_by_turbines("ti", "ti", df, turbine_numbers, False)


def set_ti_by_all_turbines(
    df: Union[pd.DataFrame, FlascDataFrame],
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add TI column using all turbines.

    Add a column called 'ti' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all
    turbines.

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
            should be used to calculate the column average.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
        plus the additional column called 'ti'.
    """
    return _set_col_by_turbines("ti", "ti", df, "all", False)


def set_ti_by_upstream_turbines(
    df: Union[pd.DataFrame, FlascDataFrame],
    df_upstream: pd.DataFrame,
    exclude_turbs: List[int] = [],
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add TI column using upstream turbines.

    Add a column called 'ti' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all the turbines
    upstream, excluding the turbines listed in exclude_turbs.

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.Dataframe): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
            exclude_turbs ([list, array]): array-like variable containing
            turbine indices that should be excluded in determining the column
            mean quantity.
        exclude_turbs ([list, array]): array-like variable containing
            turbine indices that should be excluded in determining the column
            mean quantity.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'ti'.
    """
    return _set_col_by_upstream_turbines(
        col_out="ti",
        col_prefix="ti",
        df=df,
        df_upstream=df_upstream,
        circular_mean=False,
        exclude_turbs=exclude_turbs,
    )


def set_ti_by_upstream_turbines_in_radius(
    df: Union[pd.DataFrame, FlascDataFrame],
    df_upstream: pd.DataFrame,
    turb_no: int,
    x_turbs: List[float],
    y_turbs: List[float],
    max_radius: float,
    include_itself: bool = True,
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add TI column by upstream turbines within a radius.

    Add a column called 'ti' to your dataframe, which is the
    mean of the columns ti_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.Dataframe): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
            turb_no (int): Turbine number from which the radius should be
            calculated.
        turb_no (int): Turbine number from which the radius should be
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius (float): Maximum radius for the upstream turbines
            until which they are still considered as relevant/used for the
            calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
            turb_no in the determination of the averaged column quantity. Defaults
            to False.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
        plus the additional column called 'ti'.
    """
    return _set_col_by_upstream_turbines_in_radius(
        col_out="ti",
        col_prefix="ti",
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=False,
        include_itself=include_itself,
    )


def set_pow_ref_by_turbines(
    df: Union[pd.DataFrame, FlascDataFrame], turbine_numbers: List[int]
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add power reference column by list of turbines.

    Add a column called 'pow_ref' in your dataframe with value equal
    to the averaged turbulence intensity measurements of all the
    turbines listed in turbine_numbers.

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        turbine_numbers ([list, array]): List of turbine numbers that
            should be used to calculate the column average.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'ti'.
    """
    return _set_col_by_turbines("pow_ref", "pow", df, turbine_numbers, False)


def set_pow_ref_by_upstream_turbines(
    df: Union[pd.DataFrame, FlascDataFrame],
    df_upstream: pd.DataFrame,
    exclude_turbs: List[int] = [],
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add pow_ref column using upstream turbines.

    Add a column called 'pow_ref' in your dataframe with value equal
    to the averaged power measurements of all the turbines upstream,
    excluding the turbines listed in exclude_turbs.

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.Dataframe): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
        exclude_turbs ([list, array]): array-like variable containing
            turbine indices that should be excluded in determining the column
            mean quantity.


    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'pow_ref'.
    """
    return _set_col_by_upstream_turbines(
        col_out="pow_ref",
        col_prefix="pow",
        df=df,
        df_upstream=df_upstream,
        circular_mean=False,
        exclude_turbs=exclude_turbs,
    )


def set_pow_ref_by_upstream_turbines_in_radius(
    df: Union[pd.DataFrame, FlascDataFrame],
    df_upstream: pd.DataFrame,
    turb_no: int,
    x_turbs: List[float],
    y_turbs: List[float],
    max_radius: float,
    include_itself: bool = False,
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add pow_ref column using upstream turbines within a radius.

    Add a column called 'pow_ref' to your dataframe, which is the
    mean of the columns pow_%03d for turbines that are upstream and
    also within radius [max_radius] of the turbine of interest
    [turb_no].

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.Dataframe): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
            turb_no (int): Turbine number from which the radius should be
            calculated.
        turb_no (int): Turbine number from which the radius should be
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        max_radius (float): Maximum radius for the upstream turbines
            until which they are still considered as relevant/used for the
            calculation of the averaged column quantity.
        include_itself (bool, optional): Include the measurements of turbine
            turb_no in the determination of the averaged column quantity. Defaults
            to False.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'pow_ref'.
    """
    return _set_col_by_upstream_turbines_in_radius(
        col_out="pow_ref",
        col_prefix="pow",
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        max_radius=max_radius,
        circular_mean=False,
        include_itself=include_itself,
    )


def set_pow_ref_by_n_closest_upstream_turbines(
    df: Union[pd.DataFrame, FlascDataFrame],
    df_upstream: pd.DataFrame,
    turb_no: int,
    x_turbs: List[float],
    y_turbs: List[float],
    exclude_turbs: bool = [],
    N: int = 5,
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Add pow_ref column using N-nearest upstream turbines.

    Add a column called 'pow_ref' to your dataframe, which is the
    mean of the columns pow_%03d for the N closest turbines that are
    upstream of the turbine of interest [turb_no].

    Args:
        df (pd.Dataframe | FlascDataFrame): Dataframe with measurements. This dataframe
            typically consists of wd_%03d, ws_%03d, ti_%03d, pow_%03d, and
            potentially additional measurements.
        df_upstream (pd.Dataframe): Dataframe containing rows indicating
            wind direction ranges and the corresponding upstream turbines for
            that wind direction range. This variable can be generated with
            flasc.utilities.floris_tools.get_upstream_turbs_floris(...).
        turb_no (int): Turbine number from which the radius should be
            calculated.
        x_turbs ([list, array]): Array containing x locations of turbines.
        y_turbs ([list, array]): Array containing y locations of turbines.
        exclude_turbs ([list, array]): array-like variable containing
            turbine indices that should be excluded in determining the column
            mean quantity.
        N (int): Number of closest turbines to consider for the calculation
            of the averaged column quantity.  Defaults to 5.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe which equals the inserted dataframe
            plus the additional column called 'pow_ref'.
    """
    return _set_col_by_n_closest_upstream_turbines(
        col_out="pow_ref",
        col_prefix="pow",
        N=N,
        df=df,
        df_upstream=df_upstream,
        turb_no=turb_no,
        x_turbs=x_turbs,
        y_turbs=y_turbs,
        exclude_turbs=exclude_turbs,
        circular_mean=False,
    )


def df_reduce_precision(
    df_in: Union[pd.DataFrame, FlascDataFrame],
    verbose: bool = False,
    allow_convert_to_integer: bool = True,
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Reduce dataframe precision.

    Reduce the precision in dataframes from float64 to float32, or possibly
    even further to int32, int16, int8 or even bool. This operation typically
    reduces the size of the dataframe by a factor 2 without any real loss in
    precision. This can make particular operations and data storage much more
    efficient. This can also bring about speed-ups doing calculations with
    these variables.

    Args:
        df_in (pd.Dataframe | FlascDataFrame): Dataframe that needs to be reduced.
        verbose (bool, optional): Print progress. Defaults to False.
        allow_convert_to_integer (bool, optional): Allow reduction to integer
           type if possible. Defaults to True.

    Returns:
       pd.Dataframe | FlascDataFrame: Reduced dataframe
    """
    list_out = []
    dtypes = df_in.dtypes
    for ii, c in enumerate(df_in.columns):
        datatype = str(dtypes[c])
        if (datatype == "float64") or (datatype == "float32") or (datatype == "float"):
            # Check if can be simplified as integer
            if (
                not any(np.isnan(df_in[c]))
                and allow_convert_to_integer
                and all(np.isclose(np.round(df_in[c]), df_in[c], equal_nan=True))
            ):
                unique_values = np.unique(df_in[c])
                if np.array_equal(unique_values, [0, 1]):
                    var_downsampled = df_in[c].astype(bool)
                elif np.max(df_in[c]) < np.iinfo(np.int8).max:
                    var_downsampled = df_in[c].astype(np.int8)
                elif np.max(df_in[c]) < np.iinfo(np.int16).max:
                    var_downsampled = df_in[c].astype(np.int16)
                elif np.max(df_in[c]) < np.iinfo(np.int32).max:
                    var_downsampled = df_in[c].astype(np.int32)
                else:
                    var_downsampled = df_in[c].astype(np.int64)
            else:  # If not, just simplify as float32
                var_downsampled = df_in[c].astype(np.float32)
            max_error = np.max(np.abs(var_downsampled - df_in[c]))
            if verbose:
                logger.info(
                    "Column %s ['%s'] was downsampled to %s."
                    % (c, datatype, var_downsampled.dtypes)
                )
                logger.info(f"Max error: {max_error}")
        elif (datatype == "int64") or (datatype == "int32") or (datatype == "int"):
            if np.array_equal(np.unique(df_in[c]), [0, 1]):
                var_downsampled = df_in[c].astype(bool)
            elif len(np.unique(df_in[c])) < 100:
                var_downsampled = df_in[c].astype(np.int16)
            else:
                var_downsampled = df_in[c].astype(np.int32)
            max_error = np.max(np.abs(var_downsampled - df_in[c]))
            if verbose:
                logger.info(
                    "Column %s ['%s'] was downsampled to %s."
                    % (c, datatype, var_downsampled.dtypes)
                )
                logger.info(f"Max error: {max_error}")
        else:
            if verbose:
                logger.info("Datatype '%s' not recognized. Not downsampling." % datatype)
            var_downsampled = df_in[c]

        list_out.append(var_downsampled)

    df_out = pd.concat(list_out, axis=1, ignore_index=False)
    if isinstance(df_in, FlascDataFrame):
        df_out = FlascDataFrame(df_out)
        df_out.copy_metadata(df_in)
    return df_out


# Functions used for dataframe processing specifically
def df_drop_nan_rows(
    df: Union[pd.DataFrame, FlascDataFrame], verbose: bool = False
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Drop all-nan rows.

    Remove entries in dataframe where all rows (besides 'time')
    have nan values.

    Args:
        df (pd.Dataframe | FlascDataFrame): Input pandas dataframe
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe with all-nan rows removed
    """
    N_init = df.shape[0]
    colnames = [c for c in df.columns if c not in ["time", "turbid", "index"]]
    df = df.dropna(axis=0, subset=colnames, how="all")

    if verbose:
        logger.info("Reduced dataframe from %d to %d rows." % (N_init, df.shape[0]))

    return df


def df_find_and_fill_data_gaps_with_missing(
    df: Union[pd.DataFrame, FlascDataFrame], missing_data_buffer: float = 5.0
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Find and fill data gap and mark as missing data with NaN.

    This function takes a pd.DataFrame object and look for large jumps in
       the 'time' column. Rather than simply interpolating these values using
       a ZOH, this rather indicates that measurements are missing. Hence,
       this function finds these time gaps and inserts an additional row
       extra 1 second after the start of the time gap with all 'nan' values.
       This way, the data gap becomes populated with 'nan' values and the data
       will be ignored in any further analysis.

    Args:
        df (pd.Dataframe | FlascDataFrame): Merged dataframe for all imported files
        missing_data_buffer (float, optional): If the time gaps are equal or
            larger than this limit [s], then it will consider the data as
            corrupted or missing. Defaults to 10.

    Returns:
        pd.Dataframe | FlascDataFrame: The postprocessed dataframe where all data
            within large time gaps hold value 'missing'.
    """
    df = df.sort_values(by="time")

    time_values = df["time"].values
    time_delta = np.diff(time_values)

    logger.info(
        "Largest time jump in data is %s s, from %s to %s."
        % (
            max(time_delta) / np.timedelta64(1, "s"),
            pd.to_datetime(time_values[np.where(time_delta == max(time_delta))[0][0]]),
            pd.to_datetime(time_values[np.where(time_delta == max(time_delta))[0][0] + 1]),
        )
    )
    if max(time_delta) >= np.timedelta64(datetime.timedelta(minutes=30)):
        logger.warn("Found a gap of > 30 minutes in data.\n" + " Are you missing a data file?")

    dt_buffer = np.timedelta64(missing_data_buffer)
    missing_data_idx = np.where(time_delta >= dt_buffer)[0]
    N_datagaps = len(missing_data_idx)
    td_avg = np.mean(time_delta[missing_data_idx]) / np.timedelta64(datetime.timedelta(seconds=1))
    logger.info(
        "  Found %d time jumps in data with an average of %.2f s. Filling datagaps with 'missing'."
        % (N_datagaps, td_avg)
    )
    times_to_insert = [
        pd.to_datetime(time_values[i]) + datetime.timedelta(seconds=1) for i in missing_data_idx
    ]

    # Create empty dataframe and insert times_to_insert as nans
    df_entries_missing = df[0:1].reset_index().drop(columns="index").drop(0)
    df_entries_missing["time"] = times_to_insert
    df_entries_missing = df_entries_missing.replace(pd.NaT, "missing")

    for mi in np.where(time_delta > np.timedelta64(30, "s"))[0]:
        logger.warn(
            "  Significant time jump in data of %s s has happened from %s to %s."
            % (
                time_delta[mi] / np.timedelta64(1, "s"),
                pd.to_datetime(time_values[mi]),
                pd.to_datetime(time_values[mi + 1]),
            )
        )

    df = df.append(df_entries_missing)  # Add new row with 'missing' entries
    df = df.sort_values(by="time")  # Sort by time
    df = df.reset_index().drop(columns="index")  # Reset index

    return df


def df_sort_and_find_duplicates(df: Union[pd.DataFrame, FlascDataFrame]):
    """This function sorts the dataframe and finds rows with equal time index.

    Args:
        df (pd.Dataframe | FlascDataFrame): An (unsorted) dataframe

    Returns:
        pd.Dataframe | FlascDataFrame: Dataframe sorted by time
        duplicate_entries_idx ([list of int]): list with indices of the former
            of two duplicate rows. The indices correspond to the time-sorted df.
    """
    df = df.sort_values(axis=0, by="time", ignore_index=True)
    time_delta = np.diff(df["time"].values)
    duplicate_entries_idx = np.where(np.abs(np.float64(time_delta)) < 1e-3)[0]

    # Clean up
    if "index" in df.columns:
        df = df.drop(columns="index")

    return df, duplicate_entries_idx


def is_day_or_night(
    df: Union[pd.DataFrame, FlascDataFrame],
    latitude: float,
    longitude: float,
    sunrise_altitude: float = 0,
    sunset_altitude: float = 0,
    lag_hours: float = 0,
    datetime_column: str = "time",
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Determine night or day in dataframe.

    Determine whether it's day or night for a given set of coordinates and
    UTC timestamp in a DataFrame.

    Args:
        df (pd.DataFrame | FlascDataFrame): A Pandas DataFrame containing the time in UTC
            and other relevant data.
        latitude (float): The latitude of the location for which to determine day or night.
        longitude (float): The longitude of the location for which to determine day or night.
        sunrise_altitude (float): The altitude of the sun to denote
            that sunrise has occurred [degress]
        sunset_altitude (float): The altitude of the sun to denote that
             sunset has occurred [degress]
        lag_hours (float, optional): The number of hours to lag behind the
            timestamp for the daylight
        determination. Default is 0.
        datetime_column (str, optional): The name of the DataFrame column containing
            the timestamp in UTC. Default is 'time'.

    Returns:
        pd.DataFrame | FlascDataFrame: The input DataFrame with two additional
            columns: 'sun_altitude'
            (the sun's altitude at the given timestamp)
            and 'is_day' (a boolean indicating whether it's daytime at the given timestamp).

    """
    import ephem  # Import here so don't use the memory if not calling this function

    # Create an Observer with the given latitude and longitude
    observer = ephem.Observer()

    def sun_alt(row):
        observer.lat = str(latitude)
        observer.long = str(longitude)
        observer.date = row[datetime_column] - datetime.timedelta(hours=lag_hours)
        sun = ephem.Sun()
        sun.compute(observer)
        return float(sun.alt) * 180 / np.pi

    # Add a new column 'sun_altitude' to the DataFrame
    df["sun_altitude"] = df.apply(sun_alt, axis=1)
    alt_diff = np.diff(df["sun_altitude"], prepend=0)
    alt_diff[0] = alt_diff[1]  # Assume that the first time matches the second.

    # Apply daytime criteria
    df["is_day"] = (
        ((df["sun_altitude"] > sunrise_altitude) & (alt_diff > 0))
        | ((df["sun_altitude"] > sunset_altitude) & (alt_diff < 0))
    ).astype(Boolean)

    # If a lag was provided, recompute sun_altitude at the correct time
    if lag_hours != 0:
        lag_hours = 0
        df["sun_altitude"] = df.apply(sun_alt, axis=1)

    return df


def plot_sun_altitude_with_day_night_color(
    df: Union[pd.DataFrame, FlascDataFrame], ax: plt.axis = None
) -> plt.axis:
    """Plot sun altitude with day-night color differentiation.

    This function creates a plot of Sun Altitude over time,
    distinguishing between day and night periods
    with different background colors. The input DataFrame 'df'
    should contain time and sun_altitude columns,
    as well as a boolean 'is_day' column to indicate day and night periods.

    Args:
        df (pd.DataFrame | FlascDataFrame): A DataFrame containing time, sun_altitude,
            and is_day columns.
        ax (plt.axis, optional): An optional Matplotlib axis to use for the plot.
            If not provided, a new axis will be created.

    Returns:
        ax (plt.axis): The Matplotlib axis plotted on.
    """
    # Separate the DataFrame into day and night parts
    day_data = df[df["is_day"]]
    night_data = df[~df["is_day"]]

    # Create a figure and axis for the plot
    if ax is None:
        fig, ax = plt.subplots()

    # Plot day data with a blue background
    ax.plot(
        day_data["time"],
        day_data["sun_altitude"],
        color="orange",
        label="Day",
        marker=".",
        ls="None",
    )
    # ax.fill_between(day_data['time'], day_data['sun_altitude'], color='orange', alpha=0.7)

    # Plot night data with a black background
    ax.plot(
        night_data["time"],
        night_data["sun_altitude"],
        color="darkblue",
        label="Night",
        marker=".",
        ls="None",
    )
    # ax.fill_between(night_data['time'], night_data['sun_altitude'], color='darkblue', alpha=0.7)

    # Set axis labels and a legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Sun altitude [deg]")
    ax.legend(loc="upper right")

    # Rotate x-axis labels for readability
    fig = plt.gcf()
    fig.autofmt_xdate(rotation=45)

    # Final touches
    ax.grid(True)
    ax.axhline(0, color="k", lw=2)

    return ax


def df_sort_and_fix_duplicates(
    df: Union[pd.DataFrame, FlascDataFrame],
) -> Union[pd.DataFrame, FlascDataFrame]:
    """Sort dataframe and fill duplicates.

    This function sorts the dataframe and addresses duplicate rows (i.e.,
    rows in which the time index is equal). It does this by merging the two
    rows, replacing the 'nan' entries of one row with the non-'nan' entries
    of the other row. If someone both rows have different values for the same
    column, then an exception is thrown.

    Args:
        df (pd.Dataframe | FlascDataFrame): An (unsorted) dataframe

    Returns:
        df (pd.Dataframe | FlascDataFrame): A time-sorted Dataframe in which its duplicate
        rows have been merged.
    """
    # Check and merge any duplicate entries in the dataset
    df, duplicate_time_entries = df_sort_and_find_duplicates(df)
    while len(duplicate_time_entries) > 0:
        di = duplicate_time_entries[0]
        # df_subset = df[di:di+2].copy()

        # Check if any conflicting entries exist within duplicate rows
        column_list = [c for c in df.columns if (c != "time" and c != "index")]
        df_merged = df[di : di + 1].copy().reset_index(drop=True)  # Start with first row
        for c in column_list:
            x1 = df.loc[di, c]
            x2 = df.loc[di + 1, c]

            # Check if either is NaN
            x1_isnan = not (x1 == x1)
            x2_isnan = not (x2 == x2)

            # Check if values conflict
            if x1_isnan:
                is_faulty = False
                df_merged.loc[0, c] = x2
            elif x2_isnan:
                is_faulty = False
                # Do nothing, keep x1
            elif x1 == x2:
                is_faulty = False
                # Do nothing, keep x1
            else:
                is_faulty = True
                df_merged.loc[0, c] = np.nan

            if is_faulty:
                logger.warn(
                    "Found conflicting data entries for timestamp: " + str(df.loc[di, "time"]) + "."
                )
                print(df.loc[di : di + 1, c])
                logger.info("Setting value to np.nan as a safety measure...")

        logger.info(f"Merged two rows with identical timestamp:{df.loc[di, 'time']}.")
        logger.info("Before merging:")
        print(df[di : di + 2])
        logger.info(" ")
        logger.info("After merging:")
        print(df_merged)
        logger.info(" ")

        # Now merge data
        df = df.reset_index().drop([di, di + 1])  # Remove dupl. rows
        df = df.append(df_merged)  # Add merged row

        # Sort df by 'time' and recalculate duplicate entries
        df, duplicate_time_entries = df_sort_and_find_duplicates(df)

    return df
