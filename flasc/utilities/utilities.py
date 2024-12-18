"""Utility functions for the FLASC module."""

import datetime

import numpy as np
from floris.utilities import wrap_360


def estimate_dt(time_array):
    """Automatically estimate timestep in a time_array.

    Args:
        time_array (list): List or dataframe with time entries

    Returns:
        datetime.timedelta: Timestep in dt.timedelta format
    """
    if len(time_array) < 2:
        # Assume arbitrary value
        return datetime.timedelta(seconds=0)

    dt = np.median(np.diff(time_array))
    if not isinstance(dt, datetime.timedelta):
        dt = datetime.timedelta(seconds=dt.astype(float) / 1e9)

    # Check if data is all ascending
    if dt <= datetime.timedelta(0):
        raise UserWarning("Please only insert time ascending data.")

    return dt


def get_num_turbines(df):
    """Get the number of turbines in a dataframe.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with turbine data

    Returns:
       int: Number of turbines in the dataframe
    """
    # Count how many columns in df_columns are of the form 'pow_###'
    return sum(len(c) == 7 and c[:4] == "pow_" and c[4:].isdigit() for c in df.columns)


def interp_with_max_gap(x, xp, fp, max_gap, kind, wrap_around_360=False):
    """Interpolate data linearly or using nearest-neighbor with maximum gap.

    If there is larger gap in data than `max_gap`, the gap will be filled
    with np.nan.

    Args:
        x (np.array): The output x-data; the data points in x-axis that
            you want the interpolation results from.
        xp (np.array): The input x-data.
        fp (np.array): The input y-data.
        max_gap (float): The maximum allowable distance between x and `xp` for which
            interpolation is still performed. Gaps larger than
            this will be filled with np.nan in the output `target_y`.
        kind (str): The interpolation method to use. Can be 'linear' or 'nearest'.
        wrap_around_360 (bool): If True, the interpolation will be done in a circular
            fashion, i.e., the interpolation will wrap around 360 degrees.

    Returns:
        np.array: The interpolation results.
    """
    if not ((kind == "linear") or (kind == "nearest")):
        raise NotImplementedError("Unknown interpolation method specified.")

    # Check format of max_gap: needs to be an integer/float
    if not isinstance(max_gap, (float, int)):
        max_gap = np.timedelta64(max_gap) / np.timedelta64(1, "s")

    if wrap_around_360:
        fp_cos = np.cos(fp * np.pi / 180.0)
        fp_sin = np.sin(fp * np.pi / 180.0)
        y_cos = _interpolate_with_max_gap(x, xp, fp_cos, max_gap, False, kind)
        y_sin = _interpolate_with_max_gap(x, xp, fp_sin, max_gap, False, kind)
        y = wrap_360(np.arctan2(y_sin, y_cos) * 180.0 / np.pi)
    else:
        y = _interpolate_with_max_gap(x, xp, fp, max_gap, False, kind)

    return y


# Credits to 'np8', from https://stackoverflow.com/questions/64045034/interpolate-values-and-replace-with-nans-within-a-long-gap
# Adapted to include nearest-neighbor interpolation
# @numba.njit()
def _interpolate_with_max_gap(
    x,
    xp,
    fp,
    max_gap,
    assume_sorted=False,
    kind="linear",
    extrapolate=True,
):
    """Interpolate data linearly or using nearest-neighbor with maximum gap.

    If there is larger gap in data than `max_gap`, the gap will be filled
    with np.nan.

    The input values should not contain NaNs.

    Args:
        x (np.array): The output x-data; the data points in x-axis that
            you want the interpolation results from.
        xp (np.array): The input x-data.
        fp (np.array): The input y-data.
        max_gap (float): The maximum allowable distance between x and `xp` for which
            interpolation is still performed. Gaps larger than
            this will be filled with np.nan in the output `target_y`.
        assume_sorted (bool): If True, assume that `xp` is sorted in ascending
            order. If False, sort `xp` and `fp` to be monotonous.
        kind (str): The interpolation method to use. Can be 'linear' or 'nearest'.
        extrapolate (bool): If True, extrapolate the data points on the boundaries

    Returns:
        np.array: The interpolation results.
    """
    if not assume_sorted:
        # Sort xp and fp to be monotonous
        sort_array = np.argsort(xp)
        xp = xp[sort_array]
        fp = fp[sort_array]

        # Sort x to be monotonous
        sort_array = np.argsort(x)
        inverse_sort_array = np.argsort(sort_array)  # Used to undo sort
        x = x[sort_array]

    if extrapolate:
        # Add points on boundaries for xp
        xp_full = np.empty(len(xp) + 2)
        xp_full[1:-1] = xp
        xp_full[0] = xp[0] - max_gap
        xp_full[-1] = xp[-1] + max_gap

        # Add points on boundaries for fp
        fp_full = np.empty(len(fp) + 2)
        fp_full[1:-1] = fp
        fp_full[0] = fp[0]
        fp_full[-1] = fp[-1]
    else:
        xp_full = xp
        fp_full = fp

    # # Check if we can solve it using numpy's internal interp function
    # if ((kind=='linear') and (np.max(np.diff(xp)) <= max_gap)):
    #     target_y = np.interp(x, xp, fp, left=np.nan, right=np.nan)
    #     return target_y[inverse_sort_array]

    # Otherwise, process manually: initialize variables
    target_y = np.ones(x.size) * np.nan  # Fill with NaNs
    idx_left_interp_point = 0

    # Loop through all cases to the left of minimum point xp
    ij = 0
    while x[ij] < xp_full[0]:
        ij = ij + 1  # Do nothing, leave values as nans

    # Loop through all cases that fall inside xp
    exit_loop = False
    for ii in range(ij, len(x)):
        # Move left interp point, if necessary
        while x[ii] > xp_full[idx_left_interp_point + 1]:
            idx_left_interp_point += 1
            if (idx_left_interp_point + 1) >= len(xp_full):
                # Exit, we are now to the right of max. point xp
                exit_loop = True
                break

        if exit_loop:
            break

        # Calculate coordinates to interpolate
        x1 = xp_full[idx_left_interp_point]
        y1 = fp_full[idx_left_interp_point]
        x2 = xp_full[idx_left_interp_point + 1]
        y2 = fp_full[idx_left_interp_point + 1]

        # Calculate gaps and determine if limit is exceeded
        delta_x1 = x[ii] - x1
        delta_x2 = x2 - x[ii]
        if (delta_x1 > max_gap) and (delta_x2 > max_gap):
            continue

        # Deal with when both x1 and x2 have the same value, take first one
        if x1 == x2:
            target_y[ii] = y1
            continue

        if kind == "linear":
            # Linearly interpolate over gap
            target_y[ii] = y1 + ((y2 - y1) / (x2 - x1)) * (x[ii] - x1)
        else:
            # Assume the nearest-neighbor value
            if delta_x1 > delta_x2:
                target_y[ii] = y2
            else:
                target_y[ii] = y1

    if not assume_sorted:
        return target_y[inverse_sort_array]

    return target_y
