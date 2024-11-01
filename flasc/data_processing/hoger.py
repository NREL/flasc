"""Module for homogenizing the wind direction data using the HOGER method.

HOGER was developed by Paul Poncet and Thomas Duc of Engie within the TWAIN project.

The original code was written in R (link?) amd was translated to Python by Paul Fleming.
TOOO: (1) Fact check (2) Add references (3) Add github ids
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from floris.utilities import wrap_180, wrap_360

from flasc import FlascDataFrame
from flasc.utilities.circular_statistics import calc_wd_mean_radial

# The original code contains 4 functions: modulo, mean_circ, discretize, and homogenize
# modulo: equivalent to wrap_180 (imported from FLORIS)
# mean_circ: equivalent to calc_wd_mean_radial (imported from FLASC.utilities.circular_statistics)
# discretize and homogenize are implemented below


def discretize(x: np.ndarray, threshold: float = 100) -> np.ndarray:
    """Discretize data points into segments.

    Args:
        x (np.ndarray): Data to discretize.
        threshold (float, optional): Threshold for discretization. Defaults to 100.

    Returns:
        np.ndarray: Discretized data.
    """
    # Handle NA values
    na = pd.isna(x)

    # Sort indices
    o = np.argsort(x)
    x_sorted = x[o]

    # Initialize group labels
    y = np.ones(len(x_sorted))

    # Find significant jumps
    d = np.diff(x_sorted)
    w = np.where(d >= threshold)[0]

    # Assign group labels
    for i in range(len(d)):
        if i in w:
            y[i + 1 :] += 1

    # Reorder and handle NAs
    y = y[np.argsort(o)]
    y[na] = np.nan

    return y


def homogenize(
    df: Union[pd.DataFrame, FlascDataFrame],
    threshold: float = 100,
    reference: str = "last",
    verbose: bool = False,
) -> pd.DataFrame:
    """Homegenize wind direction data using the Hoger method.
    
    Args:
        df (Union[pd.DataFrame, FlascDataFrame]): DataFrame containing the SCADA data.
        threshold (float, optional): Threshold for discretization. Defaults to 100.
        reference (str, optional): Reference point for homogenization. Defaults to 'last'.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    
    """
    # Make sure in FlascDataFrame format
    df = FlascDataFrame(df)

    # Make sure there are at least 3 turbines
    if df.n_turbines < 3:
        raise ValueError("At least 3 turbines are required for homogenization.")

    # Determine reference point
    if reference == "first":
        ref = 0
    elif reference == "last":
        ref = len(df) - 1
    else:
        try:
            ref = np.argmin(np.abs(df["time"].values - pd.to_datetime(reference)))
        except ValueError:
            raise ValueError(
                "Invalid reference point. Please use 'first', 'last', or a valid time string."
            )

    # Initialize results dataframe
    df_jump = pd.DataFrame(columns=["Knot", "Jump", "Turbine"])

    # Loop over combinations of turbines
    for t_i in range(df.n_turbines):
        t_i_col = "wd_%03d" % t_i

        if verbose:
            print(f"Processing turbine {t_i}")

        for t_j in range(df.n_turbines):
            if t_i == t_j:
                continue
            t_j_col = "wd_%03d" % t_j

            if verbose:
                print(f"...with turbine {t_j}")

            # Compute the wrapped error
            wrapped_error = wrap_180(df[t_i_col].values - df[t_j_col].values)

            # R code uses picor: Piecewise-constant regression, using
            # https://github.com/chasmani/piecewise-regression in python
            # as a replacement for picor
            # I can't find a close python equivalent for picor, so starting with ruptures
            # this is convenient as via the dependency on wind-up this is already
            # a defacto requirement for FLASC

            # Note these first lines (minus the threshold)
            # are verbatim from the example here
            # https://github.com/deepcharles/ruptures
            # presumably can improve somewhat
            algo = rpt.Pelt(model="rbf", min_size=threshold).fit(wrapped_error)
            result = algo.predict(pen=10)

            # If results is length 1 or 0, no significant jumps detected, continue
            if len(result) <= 1:
                if verbose:
                    print("... No significant jumps detected")
                continue

            if verbose:
                print(f"... Jumps detected at: {result[:-1]}")

            # Compute the mean values in error in each of the identified segments
            # so we can compute the jump size at each jump location
            knots = result[:-1]  # Exclude the end point returned by ruptures
            values = [
                calc_wd_mean_radial(wrapped_error[start:end])
                for start, end in zip([0] + knots, knots + [len(wrapped_error)])
            ]

            # Paul's note: I added wrap_180 here though I don't think it's in original R code
            # but it feels correct to me to include it since errors 
            # should not include values > abs(180)
            values = [wrap_180(v) for v in values]

            if verbose:
                print(f"... Jump values per area: {values}")

            jumps = np.diff(values)

            if verbose:
                print(f"... Jump sizes: {jumps}")

            # Append result to the result dataframe
            # TODO: Not a big deal but this is a slow way to do it
            df_jump = pd.concat(
                [df_jump, pd.DataFrame({"Knot": knots, "Jump": jumps, "Turbine": t_i})]
            )

    # Process change points
    if not df_jump.empty:
        # Group and summarize change points
        df_jump = (
            df_jump.assign(Count=1, Class=discretize(df_jump["Knot"].values, threshold=threshold))
            .groupby(["Class", "Turbine"])
            #TODO Original code uses a "mode" but for now taking a shortcut with median
            .agg(
                {
                    "Knot": "median",  # Using median instead of shorth
                    "Jump": "mean",
                    "Count": "sum",
                }
            )
            .reset_index()
            .query(f"Count > {np.floor(df.n_turbines/2)}")
            .sort_values("Count", ascending=False)
            .drop_duplicates("Class")
            .sort_values("Class")
        )

        # Apply corrections
        df_corr = df.copy()
        for _, row in df_jump.iterrows():
            m = row["Turbine"]
            k = row["Knot"]
            j = row["Jump"]

            t_col = "wd_%03d" % m

            # Simple step function approximation
            correction = np.where(np.arange(df.shape[0]) >= k, j, 0)

            # Paul note, in original form += used but -= looks better to me here
            df_corr[t_col] -= correction - correction[ref]

            # Paul note, I added this because it felt write
            df_corr[t_col] = wrap_360(df_corr[t_col])

        return df_corr
    else:
        return df


if __name__ == "__main__":
    # # Test discretize function
    # x = np.array([0, 1, 2, 3,np.nan,2, 105, 1, np.nan])
    # y = discretize(x)
    # print(y)

    # Now make a test dataframe to test the homogenize function
    # Imagine there are 3 turbines, the first turbine's wd is
    # set by a random walk.  Turbine 2 is equal to 1 + white noise
    # finally turbine 3 is turbine 1 + white noise, + a jump
    # by jump_size deg halfway through
    n = 1000
    jump_size = 10.0
    np.random.seed(0)
    time = pd.date_range("2020-01-01", periods=n, freq="10min")
    wd_000 = wrap_360(np.cumsum(np.random.randn(n)))
    wd_001 = wrap_360(wd_000 + np.random.randn(n))
    wd_002 = wd_000 + np.random.randn(n)
    wd_002[int(np.floor(n / 2)) :] += jump_size
    wd_002 = wrap_360(wd_002)

    # FlascDataFrame requires power signals, just make these up
    pow_made_up = np.random.randn(n)

    # Plot the 3 signals

    fig, ax = plt.subplots()
    ax.plot(time, wd_000, label="Turbine 0")
    ax.plot(time, wd_001, label="Turbine 1")
    ax.plot(time, wd_002, label="Turbine 2")
    ax.legend()
    ax.grid(True)
    ax.set_title("Original Wind Directions")

    # Combine into a FlascDataFrame
    df = FlascDataFrame(
        {
            "time": time,
            "wd_000": wd_000,
            "wd_001": wd_001,
            "wd_002": wd_002,
            "pow_000": pow_made_up,
            "pow_001": pow_made_up,
            "pow_002": pow_made_up,
        }
    )

    df_corr = homogenize(df, verbose=True)

    # Plot the corrected results
    fig, ax = plt.subplots()
    ax.plot(df_corr["time"], df_corr["wd_000"], label="Turbine 0")
    ax.plot(df_corr["time"], df_corr["wd_001"], label="Turbine 1")
    ax.plot(df_corr["time"], df_corr["wd_002"], label="Turbine 2")
    ax.legend()
    ax.grid(True)
    ax.set_title("Corrected Wind Directions")

    plt.show()
