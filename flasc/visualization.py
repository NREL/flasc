"""Module for visualization of FLASC data.

This module contains functions for visualizing data from the FLASC package.

"""

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt


def plot_with_wrapping(
    x,
    y,
    ax=None,
    low=0.0,
    high=360.0,
    linestyle="-",
    marker=None,
    color=None,
    label=None,
    rasterized=False,
):
    """Plot a line on an axis that deals with angle wrapping.

    Normally, using
    pyplot will blindly connects points that wrap around 360 degrees, e.g.,
    going from 357 degrees to 2 degrees. This will create a strong vertical
    line through the plot that connects the two points, while actually these
    two coordinates should be connecting by going through 360/0 deg (i.e.,
    going up from 357 deg to 360 deg and then reappearing at 0 deg, and
    connecting to 2 deg). This function automates this process and ensures
    angles are wrapped around 0/360 deg appropriately.

    Args:
        x (iteratible): NumPy array or list containing indices/time stamps of
            the data.
        y (iteratible): NumPy array containing direction/angle data that
            requires 360 deg angle wrapping. These are typically wind directions
            or nacelle headings.
        ax (plt.Axis, optional): Axis object of the matplotlib.pyplot class.
            The line will be plotted on this axis. If None specified, will create
            a figure and axis, and plot the line in there. Defaults to None.
        low (float, optional): Lower limit at which the angles should be
            wrapped. When using degrees, this should be typically 0.0 deg for wind
            directions and nacelle directions (i.e., compass directions). When using
            vane signals, this is typically -180.0 deg. When using radians,
            this should be the equivalent in radians (e.g., 0 or - np.pi).
            Defaults to 0.0.
        high (float, optional): Upper limit at which the angles should be
            wrapped. When using degrees, this should be 360.0 deg for wind
            directions and nacelle directions (i.e., compass directions).
            When using radians, this should be the equivalent in radians.
            Defaults to 360.0.
        linestyle (str, optional): Linestyle for the plot. Defaults to "-".
        marker (str, optional): Marker style for the plot. If None is
            specified, will not use markers. Defaults to None.
        color (str, optional): Color of the line and markers. Defaults to
            "black".
        label (string, optional): Label for the line and markers. If None is
            specified, will not label the line. Defaults to None.
        rasterized (bool, optional): Rasterize the plot/line and thereby remove
            its vectorized properties. This can help reduce the size of a .pdf or
            .eps file when this figure is saved, at the cost of a potential
            reduction in image quality.


    Returns:
        ax: Axis object of the matplotlib.pyplot class on which the line (and
            optionally markers) are plotted.
    """
    # Create figure, if not provided
    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        # Use matplotlib's internal color cycler
        color = ax._get_lines.prop_cycler.__next__()["color"]

    if low >= high:
        raise UserWarning("'low' must be lower than 'high'.")
    # Format inputs to numpy arrays
    x = np.array(x, copy=True)
    y = np.array(y, copy=True, dtype=float) - low  # Normalize at 0
    high_norm = high - low
    y = np.remainder(y, high_norm)

    # Initialize empty arrays
    xw = np.array(x, copy=True)[0:0]
    yw = np.array(y, copy=True)[0:0]

    # Deal with wrapping
    id_wrap_array = np.where(np.abs(np.diff(y)) > high_norm / 2.0)[0]
    id_min = 0
    for id_wrap in id_wrap_array:
        # Step size in x direction
        dx = x[id_wrap + 1] - x[id_wrap]

        # Wrap around 0 deg
        if np.diff(y)[id_wrap] > high_norm / 2.0:
            dy = y[id_wrap] - y[id_wrap + 1] + high_norm
            xtp = x[id_wrap] + dx * (y[id_wrap]) / dy  # transition point
            xw = np.hstack([xw, x[id_min : id_wrap + 1], xtp - 0.001 * dx, xtp, xtp + 0.001 * dx])
            yw = np.hstack([yw, y[id_min : id_wrap + 1], 0.0, np.nan, high_norm])

        # Wrap around 360 deg
        elif np.diff(y)[id_wrap] < -high_norm / 2.0:
            dy = y[id_wrap + 1] - y[id_wrap] + high_norm
            xtp = x[id_wrap] + dx * (high_norm - y[id_wrap]) / dy  # transition point
            xw = np.hstack([xw, x[id_min : id_wrap + 1], xtp - 0.001 * dx, xtp, xtp + 0.001 * dx])
            yw = np.hstack([yw, y[id_min : id_wrap + 1], high_norm, np.nan, 0.0])

        id_min = id_wrap + 1

    # Append remaining data
    xw = np.hstack([xw, x[id_min::]])
    yw = np.hstack([yw, y[id_min::]])

    # Reintroduce offset from 'low'
    yw = yw + low
    y = y + low

    # Now plot lines, without markers
    if marker is None:
        # Plot without marker, but with label
        ax.plot(xw, yw, linestyle=linestyle, color=color, label=label, rasterized=rasterized)
    else:
        # Plot lines, without markers
        ax.plot(xw, yw, linestyle=linestyle, color=color, rasterized=rasterized)
        # Now plot markers, only at non-transition points
        ax.scatter(x, y, marker=marker, color=color, rasterized=rasterized)

        # Now add a placeholder (empty) line with right marker for the legend
        if label is not None:
            ax.plot(
                xw[0:0],
                yw[0:0],
                linestyle=linestyle,
                marker=marker,
                label=label,
                color=color,
                rasterized=rasterized,
            )

    return ax


def generate_default_labels(fm):
    """Generate default labels for a FlorisModel.

    Args:
        fm (FlorisModel): A FlorisModel instance.

    Returns:
        list: A list of labels for the turbines in the FlorisModel.
    """
    labels = ["T{0:02d}".format(ti) for ti in range(len(fm.layout_x))]
    return labels


def generate_labels_with_hub_heights(fm):
    """Generate labels for a FlorisModel with hub heights.

    Args:
        fm (FlorisModel): A FlorisModel instance.

    Returns:
        list: A list of labels for the turbines in the FlorisModel.
    """
    labels = [
        "T{0:02d} ({1:.1f} m)".format(ti, h)
        for ti, h in enumerate(fm.core.farm.hub_heights.flatten())
    ]
    return labels


def plot_binned_mean_and_ci(
    x,
    y,
    color="b",
    label="_nolegend_",
    x_edges=None,
    ax=None,
    show_scatter=True,
    show_bin_points=True,
    show_confidence=True,
    alpha_scatter=0.1,
    confidence_level=0.95,
):
    """Plot the mean and confidence interval of y as a function of x.

    Method has options to include scatter of underlying data, specifying
    bin edges, and plotting confidence interval.

    Args:
        x (np.array): abscissa data.
        y (np.array): ordinate data.
        color (str, optional): line color.
            Defaults to 'b'.
        label (str, optional): line label used in legend.
            Defaults to '_nolegend_'.
        x_edges (np.array, optional): bin edges in x data
            Defaults to None.
        ax (:py:class:`matplotlib.pyplot.axes`, optional):
            axes handle for plotting. Defaults to None.
        show_scatter (bool, optional): flag to control scatter plot.
            Defaults to True.
        show_bin_points (bool, optional): flag to control plot of bins.
            Defaults to True.
        show_confidence (bool, optional): flag to control plot of
            confidence interval. Defaults to True.
        alpha_scatter (float, optional): Alpha for scatter
            plot. Defaults to 0.5.
        confidence_level (float, optional): Confidence level for
            confidence interval. Defaults to 0.95.

    """
    # Check the length of x equals length of y
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")

    # Check that x is not empty
    if len(x) == 0:
        raise ValueError("x is empty")

    # Declare ax if not provided
    if ax is None:
        _, ax = plt.subplots()

    # Put points ino dataframe
    df = pd.DataFrame({"x": x, "y": y})

    # If x_edges not provided, use 50 bins over range of x
    if x_edges is None:
        x_edges = np.linspace(df["x"].min() * 0.98, df["x"].max() * 1.02, 50)

    # Define x_labels as bin centers
    x_labels = (x_edges[1:] + x_edges[:-1]) / 2.0

    # Bin data
    df["x_bin"] = pd.cut(df["x"], x_edges, labels=x_labels)

    # Get aggregate statistics
    df_agg = df.groupby("x_bin").agg({"y": ["count", "std", "min", "max", "mean", st.sem]})
    # Flatten column names
    df_agg.columns = ["_".join(c) for c in df_agg.columns]

    # Reset the index
    df_agg = df_agg.reset_index()

    # Delete rows with no data
    df_agg = df_agg[df_agg["y_count"] > 0]

    # Add the confidence interval of the mean to df_agg
    valid_sem = df_agg["y_sem"] > 0
    y_ci_lower, y_ci_upper = st.t.interval(
        confidence_level,
        df_agg[valid_sem]["y_count"] - 1,
        loc=df_agg[valid_sem]["y_mean"],
        scale=df_agg[valid_sem]["y_sem"],
    )
    df_agg["y_ci_lower"] = np.nan
    df_agg["y_ci_upper"] = np.nan
    df_agg.loc[valid_sem, "y_ci_lower"] = y_ci_lower
    df_agg.loc[valid_sem, "y_ci_upper"] = y_ci_upper

    # Plot the mean values
    ax.plot(df_agg.x_bin, df_agg.y_mean, color=color, label=label)

    # Plot the confidence interval
    if show_confidence:
        ax.fill_between(
            df_agg.x_bin,
            df_agg.y_ci_lower,
            df_agg.y_ci_upper,
            color=color,
            alpha=0.2,
        )

        # Plot a dasshed line at confidence interval
        ax.plot(
            df_agg.x_bin,
            df_agg.y_ci_lower,
            color=color,
            alpha=0.2,
            ls="--",
        )
        ax.plot(
            df_agg.x_bin,
            df_agg.y_ci_upper,
            color=color,
            alpha=0.2,
            ls="--",
        )

    # Plot the scatter points
    if show_scatter:
        ax.scatter(df.x, df.y, color=color, s=10, alpha=alpha_scatter)

    # Plot the bin points, scaled by the counts
    if show_bin_points:
        ax.scatter(
            df_agg.x_bin,
            df_agg.y_mean,
            color=color,
            s=df_agg.y_count / df_agg.y_count.max() * 20,
            alpha=0.5,
            marker="s",
        )

    return ax
