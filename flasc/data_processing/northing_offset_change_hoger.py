"""Module for homogenizing the wind direction data using the HOGER method.

HOGER was developed by Paul Poncet (https://github.com/engie-paul-poncet)
 and Thomas Duc (https://github.com/engie-thomas-duc) of Engie,
and Rubén González-Lope (https://github.com/rglope) and
Alvaro Gonzalez Salcedo (https://github.com/alvarogonzalezsalcedo) of
CENER within the TWAIN project.
"""

from __future__ import annotations

import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floris.utilities import wrap_180
from matplotlib import dates
from scipy.interpolate import interp1d
from sklearn.tree import DecisionTreeRegressor

from flasc import FlascDataFrame

_MODE_LIMIT = 0.05


def _get_leaves_and_knots(tree: DecisionTreeRegressor) -> tuple[np.ndarray, np.ndarray]:
    """Function to get the values of the superficial knots and leaves of a Tree Regression.

    Args:
        tree (DecisionTreeRegressor): Decision Tree Regression model.

    Returns:
        tuple[np.ndarray, np.ndarray]: Values of the leaves and positions of the knots.
    """
    # Get the main information from the tree
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    # Explore the results to extract the values of the leaves
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()

        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack` so we can loop t
        # through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # Get the values and the position of the external knots
    leave_values = values[is_leaves][:, 0, 0]
    knot_positions = np.sort(threshold[~is_leaves])

    return leave_values, knot_positions


def _discretize(x: pd.Series, threshold: int) -> np.ndarray:
    """Get the class of the knots based on the times they repeat.

    Args:
        x (pd.Series): Series of knot positions of the trees for the different wind turbines.
        threshold (int): Threshold used to declare a tree branch.

    Returns:
        np.ndarray: Classes of the knots.
    """
    o = np.argsort(x)  # Get the order of the knots
    x = x.iloc[o]  # Order the knots in ascending value of their positions
    y = np.ones(len(x), dtype=np.int32)
    d = np.diff(x)
    w = np.where(d >= threshold)[0]

    for i in range(len(d)):
        y[i + 1] = y[i] + (i in w)

    y = y[np.argsort(o)]

    return y


def _shorth_mode(x: pd.Series) -> np.float64:
    """Estimates the Venter mode through the shorth method for the given data.

    Args:
        x (pd.Series): Data for which the mode will be estimated

    Returns:
        np.float64: Mode of the data

    """
    ny = len(x)
    k = int(np.ceil(ny / 2) - 1)
    y = np.sort(x)
    diffs = y[k:] - y[: (ny - k)]
    i = np.where(diffs == min(diffs))[0]

    if len(i) > 1:
        if (np.max(i) - np.min(i)) > (_MODE_LIMIT * ny):
            warnings.warn(
                "Encountered a tie, and the difference between minimal and maximal value "
                f"is > length('x') * {_MODE_LIMIT}.\n The distribution could be multimodal"
            )
        i = int(np.mean(i))
    else:
        i = i[0]

    mode = np.mean(y[i : (i + k + 1)])

    return mode


def _plot_regression(y_data: pd.Series, y_regr: np.ndarray, date_time: pd.Series, ylabel: str):
    """Function to plot the results of the regression tree.

    Args:
        y_data (pd.Series): Data used on the regression.
        y_regr (np.ndarray): Results obtained from the tree regression.
        date_time (pd.Series): Dates of the original data.
        ylabel (str): Data that is shown in the plot.

    """
    fig = plt.figure()
    sc = plt.scatter(date_time, y_data, c=dates.date2num(date_time), s=5)
    cbar = fig.colorbar(sc)
    loc = dates.AutoDateLocator()
    cbar.ax.yaxis.set_major_locator(loc)
    cbar.ax.yaxis.set_major_formatter(dates.ConciseDateFormatter(loc))
    plt.plot(date_time[~y_data.isna()], y_regr, c="tab:red")
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()

    plt.show()


def homogenize_hoger(
    scada: Union[pd.DataFrame | FlascDataFrame],
    var: str = "wd",
    threshold: int = 1000,
    reference: str = "last",
    plot_it: bool = False,
    max_depth: int = 4,
    ccp_alpha: float = 0.09,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Homogenization routine of the Scada directions of the different wind turbines based on "var".

    The Scada data is explored by applying a regression tree procedure to the differences
    in direction nof the wind turbines
    to get the most common jumps and their positions.
    These jumps are then reversed to correct the deviations.

    Args:
        scada (Union[pd.DataFrame, FlascDataFrame]): DataFrame containing the SCADA data.
        var (str, optional): Variable to homogenize (yaw or wd). Defaults to 'wd'.
        threshold (int, optional): Threshold for discretization. Defaults to 1000.
        reference (str, optional): Reference point for homogenization. Defaults to 'last'.
        plot_it (bool, optional): Whether to plot the results. Defaults to False.
        max_depth (int, optional): Maximum depth of the regression tree. Defaults to 4.
        ccp_alpha (float, optional): Complexity parameter for pruning. Defaults to 0.09

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Homogenized SCADA data and the results used to
            homogenize it with the jumps and their knots.
    """
    # Check the variable to use for the homogenization
    if var not in ["yaw", "wd"]:
        raise ValueError(
            'Please, select a valid variable to use during homogeneization: "yaw" or "wd".'
        )

    # Select the columns to use in the algorithm
    wt_names = scada.columns[scada.columns.str.startswith((var + "_"))]
    if len(wt_names) < 3:
        raise ValueError("There must be at least 3 wind turbines for the algorithm to apply.")
    df = scada[wt_names.to_list() + ["time"]].reset_index(drop=True)

    # Reference date
    if reference == "first":
        ref = 0
    elif reference == "last":
        ref = df.shape[0] - 1
    else:
        date_ref = pd.to_datetime(reference)
        ref = df["time"][df["time"] == date_ref]
        if len(ref) == 0:
            if date_ref < df["time"].min():
                ref = 0
            elif date_ref > df["time"].max():
                ref = df.shape[0] - 1
            else:
                ref = 0
                warnings.warn(
                    "The reference date seem to be missing in the dataset. "
                    " The first date is selected as reference."
                )

    # Build the DataFrame to store the results of the Tree
    d = pd.DataFrame(columns=["Knot", "Jump", "Turbine"])
    d = d.astype({"Knot": np.float64, "Jump": np.float64, "Turbine": str})  # Assign types

    # Iterate over every wind turbine comparing its direction with every other.
    #  Get then the points at which
    # a jump is produced. These points are called knots.
    for m in wt_names:
        # Get the wind turbines to compare
        ms2 = wt_names[wt_names != m]
        df2 = df[ms2]
        for m2 in ms2:
            # Get the differences in the direction
            df2.loc[:, m2] = wrap_180(df[m2] - df[m])
            y = df2[m2][~df2[m2].isna()]  # Do not use the nan values
            # Use a decision tree regressor to get the points at which there are knots
            # and the values of the
            # direction differences
            regr = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=threshold,
                min_samples_leaf=threshold // 2,
                ccp_alpha=ccp_alpha,
            )
            regr.fit(y.index.to_numpy()[:, np.newaxis], y)

            # Postprocess the results of the decision tree
            # regressor to get the information required.
            # The leaves are
            # the values of the means and the knots the points at which jumps occur
            leaves, knots = _get_leaves_and_knots(regr)

            # Plot the results if desired
            if plot_it:
                _plot_regression(
                    df2[m2],
                    regr.predict(y.index.to_numpy()[:, np.newaxis]),
                    scada.time,
                    f"{m2} - {m} wind direction [°]",
                )

            # Save the results
            if len(knots) > 0 and not np.any(np.isnan(knots)):
                if m == "wd_004":
                    print(knots)
                d = pd.concat(
                    [d, pd.DataFrame({"Knot": knots, "Jump": np.diff(leaves), "Turbine": m})]
                )

    d = d.reset_index(drop=True)

    # Postprocess all the data to get the main jumps for each wind turbine
    d2 = d.copy()
    d2["Class"] = _discretize(d["Knot"], threshold=threshold // 2)
    d2["Count"] = 1
    d2 = d2.groupby(["Class", "Turbine"]).agg(
        {"Count": "sum", "Jump": "mean", "Knot": _shorth_mode}
    )
    d2.reset_index(drop=False, inplace=True)
    d2["Knot_date"] = df["time"].values[np.floor(d2["Knot"]).astype(int) - 1]
    d2 = d2.loc[d2["Count"] > len(wt_names) / 2]
    d2.sort_values("Count", ascending=False, inplace=True)
    d2.drop_duplicates(subset="Class", inplace=True)
    d2.sort_values("Class", inplace=True)
    d2.reset_index(drop=True, inplace=True)

    # Predict
    if d.shape[0] > 0:
        for i in range(d2.shape[0]):
            m = d2["Turbine"][i]
            k = d2["Knot"][i]
            j = d2["Jump"][i]
            # Build a piecewise function based on the knot and the jump
            f = interp1d(
                np.array([0, k, scada.index.max()]), np.array([0.0, j, j]), kind="previous"
            )

            scada[m] = (scada[m] + f(scada.index) - f(ref)) % 360

    return scada, d2
