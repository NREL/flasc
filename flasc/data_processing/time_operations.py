"""Time operations for data processing."""

from datetime import timedelta as td

import numpy as np
import pandas as pd
from floris.utilities import wrap_360

from flasc.logging_manager import LoggingManager
from flasc.utilities import utilities as fsut

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def df_movingaverage(
    df_in,
    cols_angular,
    window_width=td(seconds=60),
    min_periods=1,
    center=True,
    calc_median_min_max_std=False,
):
    """Compute a moving average of a dataframe with angular columns.

    Note that median, minimum, and maximum do not handle angular
    quantities and should be treated carefully.
    Standard deviation handles angular quantities.

    Args:
        df_in (pd.DataFrame | FlascDataFrame): Input dataframe.
        cols_angular (list): List of angular columns.
        window_width (timedelta): Width of the moving average window.
        min_periods (int): Minimum number of periods to consider.
        center (bool): Center the time index.  Default is True.
        calc_median_min_max_std (bool): Calculate median, min, max, and std.
            Default is False.

    Returns:
        pd.DataFrame: Output dataframe with moving averages.

    """
    df = df_in.set_index("time").copy()

    # Find non-angular columns
    if isinstance(cols_angular, bool):
        if cols_angular:
            cols_angular = [c for c in df.columns]
        else:
            cols_angular = []
    cols_regular = [c for c in df.columns if c not in cols_angular]

    # Save the full columns
    full_columns = df.columns

    # Carry out the mean calculations
    df_regular = (
        df[cols_regular]  # Select only non-angular columns
        .rolling(window_width, center=center, axis=0, min_periods=min_periods)
        .mean()
    )

    df_cos = (
        df[cols_angular]  # Select only angular columns
        .pipe(lambda df_: np.cos(df_ * np.pi / 180.0))
        .rolling(window_width, center=center, axis=0, min_periods=min_periods)
        .mean()
    )

    df_sin = (
        df[cols_angular]  # Select only angular columns
        .pipe(lambda df_: np.sin(df_ * np.pi / 180.0))
        .rolling(window_width, center=center, axis=0, min_periods=min_periods)
        .mean()
    )

    dfm = df_regular.join((np.arctan2(df_sin, df_cos) * 180.0 / np.pi) % 360)[
        full_columns
    ]  # put back in order

    if not calc_median_min_max_std:
        return dfm

    if calc_median_min_max_std:  # if including other statistics
        df_regular_stats = (
            df.rolling(window_width, center=center, axis=0, min_periods=min_periods)
            .agg(["median", "min", "max", "std"])
            .pipe(lambda df_: flatten_cols(df_))
        )

        # Apply scipy.stats.circstd() step by step for performance reasons
        df_angular_std = (
            df_sin.pow(2)
            .add(df_cos.pow(2))
            .pow(1 / 2)  # sqrt()
            .apply(np.log)  # log()
            .mul(-2)
            .pow(1 / 2)  # sqrt()
            .mul(180 / np.pi)
            .rename({c: c + "_std" for c in dfm.columns}, axis="columns")
        )

        # Merge the stats
        df_stats = df_regular_stats[
            [c for c in df_regular_stats.columns if c not in df_angular_std.columns]
        ].join(df_angular_std)

        # Now merge in means and return
        return dfm.rename({c: c + "_mean" for c in dfm.columns}, axis="columns").join(df_stats)


def df_downsample(
    df_in,
    cols_angular,
    window_width=td(seconds=60),
    min_periods=1,
    center=False,
    calc_median_min_max_std=False,
    return_index_mapping=False,
):
    """Downsample a dataframe to a average accounting for angular columns.

    Args:
        df_in (pd.DataFrame | FlascDataFrame): Input dataframe.
        cols_angular (list): List of angular columns.
        window_width (timedelta): Width of the average window.
        min_periods (int): Minimum number of data points for a bin to be valid.
        center (bool): Center the time index.  Default is False.
        calc_median_min_max_std (bool): Calculate median, min, max, and std.
            Default is False.
        return_index_mapping (bool): Return index mapping.  Default is False.

    Returns:
        A tuple (pd.DataFrame, np.ndarray) if return_index_mapping is True.  Where
            the DataFrame is the downsampled dataframe and the np.ndarray is the
            index mapping.
        A pd.DataFrame | FlascDataFrame if return_index_mapping is False.

    """
    # Copy and ensure dataframe is indexed by time
    df = df_in.copy()
    if "time" not in df.columns:
        raise KeyError('"time" must be in df_in columns.')
    df = df.set_index("time")

    # Find non-angular columns
    cols_regular = [c for c in df.columns if c not in cols_angular]

    # Now calculate cos and sin components for angular columns
    sin_cols = ["{:s}_sin".format(c) for c in cols_angular]
    cos_cols = ["{:s}_cos".format(c) for c in cols_angular]

    # Add in the sin/cos columns
    df = pd.concat(
        [df, np.sin(df[cols_angular] * np.pi / 180.0).set_axis(sin_cols, axis=1)], axis=1
    )
    df = pd.concat(
        [df, np.cos(df[cols_angular] * np.pi / 180.0).set_axis(cos_cols, axis=1)], axis=1
    )

    # Drop angular columns
    df = df.drop(columns=cols_angular)

    # Add _N for each variable to keep track of n.o. data points
    cols_all = df.columns
    cols_N = ["{:s}_N".format(c) for c in cols_all]
    df = pd.concat([df, 1 - df[cols_all].isna().astype(int).set_axis(cols_N, axis=1)], axis=1)

    # Now calculate downsampled dataframe, automatically
    # mark by label on the right (i.e., "past 10 minutes").
    df_resample = df.resample(window_width, label="right", axis=0)

    # First calculate mean values of non-angular columns
    df_mean = df_resample[cols_regular].mean().copy()

    # Compute and append the angular means
    df_mean = pd.concat(
        [
            df_mean,
            pd.DataFrame(
                wrap_360(
                    np.arctan2(
                        df_resample[sin_cols].mean().values, df_resample[cos_cols].mean().values
                    )
                    * 180.0
                    / np.pi
                ),
                columns=cols_angular,
                index=df_mean.index,
            ),
        ],
        axis=1,
    )

    # Check if we have enough samples for every measurement
    if min_periods > 1:
        N_counts = df_resample[cols_N].sum()
        df_mean[N_counts < min_periods] = None  # Remove data relying on too few samples

    # Calculate median, min, max, std if necessary
    if calc_median_min_max_std:
        df_stats = df_in.copy().set_index("time")

        # Compute the stats for the non_angular columns
        df_stats_regular = (
            df_stats[cols_regular]  # Select non-angular columns
            .resample(window_width, label="right", axis=0)  # Resample to desired window
            .agg(["median", "min", "max", "std"])  # Perform aggregations
            .pipe(lambda df_: flatten_cols(df_))  # Flatten columns
        )

        # Now to compute the statistics for the angular columns, which requires
        # shifting by the mean values
        df_angular_mean_upsample = (
            df_mean[cols_angular]  # Select angular columns
            .reindex(df_stats.index)  # Go back to original time index
            .bfill()  # Back fill the points since using right indexing
            .ffill()  # Cover any stragglers at end
        )

        df_angular_stats = (
            df_stats[cols_angular]
            .subtract(df_angular_mean_upsample)  # Subtract the angular mean
            .add(180)  # Shift up by 180 (start of sequence for -180/180 wrap)
            .mod(360)  # Wrap by 360
            .subtract(180)  # Remove shift (end of sequence for -180/180 wrap)
            .resample(window_width, label="right", axis=0)  # Resample to desired window
        )

        # Now create the individual statistics
        df_angular_median = (
            df_angular_stats.median()  # Apply the median
            .add(df_mean[cols_angular])  # Shift back by original mean
            .mod(360)  # Wrap by 360
            .rename({c: "%s_median" % c for c in cols_angular}, axis="columns")
        )

        df_angular_min = (
            df_angular_stats.min()  # Apply the min
            .add(df_mean[cols_angular])  # Shift back by original mean
            .mod(360)  # Wrap by 360
            .rename({c: "%s_min" % c for c in cols_angular}, axis="columns")
        )

        df_angular_max = (
            df_angular_stats.max()  # Apply the max
            .add(df_mean[cols_angular])  # Shift back by original mean
            .mod(360)  # Wrap by 360
            .rename({c: "%s_max" % c for c in cols_angular}, axis="columns")
        )

        # Apply scipy.stats.circstd() step by step for performance reasons
        df_angular_std = (
            df_resample[sin_cols]  # Get sine columns
            .mean()  # Apply mean
            .rename(
                {"{:s}_sin".format(c): "{:s}_std".format(c) for c in cols_angular}, axis="columns"
            )  # Rename for add()
            .pow(2)
            .add(
                df_resample[cos_cols]
                .mean()
                .rename(
                    {"{:s}_cos".format(c): "{:s}_std".format(c) for c in cols_angular},
                    axis="columns",
                )
                .pow(2)
            )  # Now have mean(sin(wd))**2 + mean(cos(wd))**2
            .pow(1 / 2)  # sqrt()
            .apply(np.log)  # log()
            .mul(-2)
            .pow(1 / 2)  # sqrt()
            .mul(180 / np.pi)
        )

        # df_out is the concatination of all these matrices
        df_out = pd.concat(
            [
                df_mean.rename({c: "%s_mean" % c for c in df_mean.columns}, axis="columns"),
                df_stats_regular,
                df_angular_median,
                df_angular_min,
                df_angular_max,
                df_angular_std,
            ],
            axis=1,
        )

        df_out = df_out[sorted(df_out.columns)]

    else:  # if not computing stats
        df_out = df_mean

    if center:
        # Shift time column towards center of the bin
        df_out.index = df_out.index - window_width / 2.0

    if return_index_mapping:
        df_tmp = pd.DataFrame(data={"time": df.reset_index()["time"], "tmp": 1}).resample(
            window_width, on="time", label="right", axis=0
        )

        # Grab index of first and last time entry for each window
        def get_first_index(x):
            if len(x) <= 0:
                return -1
            else:
                return x.index[0]

        def get_last_index(x):
            if len(x) <= 0:
                return -1
            else:
                return x.index[-1]

        windows_min = df_tmp.apply(get_first_index)["tmp"].to_list()
        windows_max = df_tmp.apply(get_last_index)["tmp"].to_list()

        # Now create a large array that contains the array of indices, with
        # the values in each row corresponding to the indices upon which that
        # row's moving/rolling average is based. Note that we purposely create
        # a larger matrix than necessary, since some rows/windows rely on more
        # data (indices) than others. This is the case e.g., at the start of
        # the dataset, at the end, and when there are gaps in the data. We fill
        # the remaining matrix entries with "-1".
        dn = int(np.ceil(window_width / fsut.estimate_dt(df_in["time"]))) + 5
        data_indices = -1 * np.ones((df_out.shape[0], dn), dtype=int)
        for ii in range(len(windows_min)):
            lb = windows_min[ii]
            ub = windows_max[ii]
            if not ((lb == -1) | (ub == -1)):
                ind = np.arange(lb, ub + 1, dtype=int)
                data_indices[ii, ind - lb] = ind

        return df_out.reset_index(), data_indices
    else:
        return (
            df_out.reset_index()
        )  # Conform to new standard that, between functions, time is a column


def df_resample_by_interpolation(
    df, time_array, circular_cols, interp_method="linear", max_gap=None, verbose=True
):
    """Resample a dataframe by interpolation onto a new time array.

    Args:
        df (pd.DataFrame | FlascDataFrame): Input dataframe.
        time_array (np.array): New time array.
        circular_cols (list): List of columns that are circular.
        interp_method (str): Interpolation method.  Default is "linear".
        max_gap (float): Maximum gap for interpolation.  Default is None.
            If None, will be set to 1.5 times the median timestep.
        verbose (bool): Print information.  Default is True.

    Returns:
        pd.DataFrame | FlascDataFrame: Resampled dataframe

    """
    # Copy with properties but no actual data
    df_res = df.head(0).copy()

    # Remove timezones, if any
    df = df.copy()
    time_array = [pd.to_datetime(t).tz_localize(None) for t in time_array]
    time_array = np.array(time_array, dtype="datetime64")
    df["time"] = df["time"].dt.tz_localize(None)

    # Fill with np.nan values and the correct time array (without tz)
    df_res["time"] = time_array

    t0 = time_array[0]
    df_t = np.array(df["time"] - t0, dtype=np.timedelta64)
    xp = df_t / np.timedelta64(1, "s")  # Convert to regular seconds
    xp = np.array(xp, dtype=float)

    # Normalize time variables
    time_array = np.array([t - t0 for t in time_array], dtype=np.timedelta64)
    x = time_array / np.timedelta64(1, "s")

    if max_gap is None:
        max_gap = 1.5 * np.median(np.diff(x))
    else:
        max_gap = np.timedelta64(max_gap) / np.timedelta64(1, "s")

    cols_to_interp = [c for c in df_res.columns if c not in ["time"]]

    # NN interpolation: just find indices and map accordingly for all cols
    for ii, c in enumerate(cols_to_interp):
        if isinstance(circular_cols, bool):
            wrap_around_360 = circular_cols
        elif isinstance(circular_cols[0], bool):
            wrap_around_360 = circular_cols[ii]
        elif isinstance(circular_cols[0], str):
            wrap_around_360 = c in circular_cols

        dt_raw_median = df["time"].diff().median() / td(seconds=1)
        if verbose:
            logger.info(
                f"  Resampling column '{c}' with median timestep {dt_raw_median:.3f} s "
                f"onto a prespecified time array with kind={interp_method}, max_gap={max_gap}"
                f"s, and wrap_around_360={wrap_around_360}"
            )

        fp = np.array(df[c], dtype=float)
        ids = (~np.isnan(xp)) & ~(np.isnan(fp))

        y = fsut.interp_with_max_gap(
            x=x,
            xp=xp[ids],
            fp=fp[ids],
            max_gap=max_gap,
            kind=interp_method,
            wrap_around_360=wrap_around_360,
        )
        df_res[c] = y

    return df_res


# Function from "EFFECTIVE PANDAS" for flattening multi-level column names
def flatten_cols(df):
    """Flatten multi-level columns in a DataFrame.

    Args:
        df (pd.DataFrame | FlascDataFrame): Input DataFrame.

    Returns:
        pd.DataFrame | FlascDataFrame: Flattened DataFrame.

    """
    cols = ["_".join(map(str, vals)) for vals in df.columns.to_flat_index()]
    df.columns = cols
    return df
