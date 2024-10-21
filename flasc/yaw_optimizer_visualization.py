"""Module for visualizing yaw optimizer results."""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flasc.logging_manager import LoggingManager

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def plot_uplifts_by_atmospheric_conditions(
    df_list,
    labels=None,
    ws_edges=np.arange(3.0, 17.0, 1.0),
    wd_edges=np.arange(0.0, 360.0001, 3.0),
    ti_edges=np.arange(0.0, 0.30, 0.02),
):
    """Plot relative power gains and contributions to AEP uplift by atmospheric conditions.

    This function plots the relative power gains and contributions to AEP uplift

    Args:
        df_list (List[pd.DataFrame | FlascDataFrame]): List of dataframes with power gains
            and contributions to AEP uplift.
        labels (List[str]): List of labels for the dataframes. Defaults to None.
        ws_edges (np.array): Wind speed bin edges. Defaults to np.arange(3.0, 17.0, 1.0).
        wd_edges (np.array): Wind direction bin edges. Defaults to np.arange(0.0, 360.0001, 3.0).
        ti_edges (np.array): Turbulence intensity bin edges. Defaults to np.arange(0.0, 0.30, 0.02).
    """
    # Calculate bin means
    ws_labels = (ws_edges[0:-1] + ws_edges[1::]) / 2.0
    wd_labels = (wd_edges[0:-1] + wd_edges[1::]) / 2.0
    ti_labels = (ti_edges[0:-1] + ti_edges[1::]) / 2.0

    # Format input
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    # Clean dataframes and calculate AEP gains
    for ii, df in enumerate(df_list):
        # Only keep cases with Pbl > 0 and non-NaNs
        df = df.dropna(how="any", subset=["farm_power_baseline", "farm_power_opt"])
        df = df[df["farm_power_baseline"] > 0.01].reset_index(drop=True)

        # Check if frequency vector exists
        if "farm_energy_baseline" not in df.columns:
            if "frequency" in df.columns:
                df["frequency"] = df["frequency"].astype(float)
            elif "freq" in df.columns:
                df["frequency"] = df["freq"].astype(float)
            else:
                df["frequency"] = 1.0
                logger.info(
                    "No column 'freq' or 'frequency' found in dataframe."
                    + "Assuming a uniform distribution."
                )

            # Calculate wind farm energy, baseline and optimized
            df["farm_energy_baseline"] = df["farm_power_baseline"] * df["frequency"]
            df["farm_energy_opt"] = df["farm_power_opt"] * df["frequency"]

        # Calculate relative and absolute uplift in energy for every condition
        df["Prel"] = np.where(
            df["farm_power_baseline"] > 0.0, df["farm_power_opt"] / df["farm_power_baseline"], 0.0
        )

        df["Pabs"] = (df["farm_energy_opt"] - df["farm_energy_baseline"]) / np.nansum(
            df["farm_energy_opt"] - df["farm_energy_baseline"]
        )

        # Bin data by wind speed, wind direction and turbulence intensity
        df["ws_bin"] = pd.cut(df["wind_speed"], ws_edges, right=False, labels=ws_labels)
        df["wd_bin"] = pd.cut(df["wind_direction"], wd_edges, right=False, labels=wd_labels)
        df["ti_bin"] = pd.cut(df["turbulence_intensity"], ti_edges, right=False, labels=ti_labels)

        df_list[ii] = df.copy()  # Save updated dataframe to self

    for yii, yq_col in enumerate(["Prel", "Pabs"]):
        if yii == 0:
            ylabel = "Relative power gain (%)"
        else:
            ylabel = "Contribution to AEP uplift (%)"

        for xii, xq_col in enumerate(["ws_bin", "wd_bin", "ti_bin"]):
            if xii == 0:
                xlabel = "Wind speed (m/s)"
            elif xii == 1:
                xlabel = "Wind direction (deg)"
            elif xii == 2:
                xlabel = "Turbulence intensity (%)"
                if np.all([df["turbulence_intensity"].unique() <= 1 for df in df_list]):
                    # Skip TI, if only optimized and evaluated for single TI
                    break

            # Now produce plots with dataframes: wind speed vs. relative power gain
            x = [None for _ in range(len(df_list))]
            y = [None for _ in range(len(df_list))]
            f = [None for _ in range(len(df_list))]
            for dii, df in enumerate(df_list):
                df_group = df.groupby(xq_col)
                if yii == 0:
                    y[dii] = 100.0 * (df_group["Prel"].apply(np.nanmean) - 1.0)
                else:
                    y[dii] = 100.0 * (df_group["Pabs"].apply(np.nansum))
                f[dii] = df_group["frequency"].apply(np.nansum)
                x[dii] = np.array(y[dii].index, dtype=float)

            fig, ax = _plot_bins(x, y, f, xlabel, ylabel, labels)


def _plot_bins(x, y, yn, xlabel=None, ylabel=None, labels=None):
    # Assume x, y and yn are lists of lists
    if isinstance(x[0], (float, int)):
        x = [x]
        y = [y]
        yn = [yn]

    # Get number of dataframes to plot
    nd = len(x)

    if labels is None:
        labels = [None for _ in range(nd)]

    # Produce plots
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 4))
    if np.all([len(xi) <= 1 for xi in x]):
        dx = 1.0  # Default to 1.0 bin width, if only one value
    else:
        dx = np.min(np.hstack([np.diff(xi) for xi in x])) * 0.8 / nd  # Bin width

    for dii in range(nd):
        # Produce top subplot
        ax[0].bar(x=x[dii] + (dii - 0.5) * dx, height=y[dii], width=dx, label=labels[dii])
        ax[0].set_ylabel(ylabel)
        ax[0].grid(True)

        ax[1].bar(x=x[dii] + (dii - 0.5) * dx, height=yn[dii], width=dx, label=labels[dii])
        ax[1].set_ylabel("Frequency (-)")
        ax[1].set_xlabel(xlabel)
        ax[1].grid(True)
        ax[1].set_xticks(x[dii])
        if len(x[dii]) > 50:  # Too many ticks: reduce
            xtlabels = ["" for _ in range(len(x[dii]))]
            xtlabels[0::5] = ["%.1f" % i for i in x[dii][0::5]]
        else:
            xtlabels = ["%.1f" % i for i in x[dii]]
        ax[1].set_xticklabels(xtlabels)

    if not np.all([a is None for a in labels]):
        ax[0].legend()
        ax[1].legend()

    return fig, ax


def plot_offsets_wswd_heatmap(df_offsets, turb_id, ax=None):
    """Plot offsets for a single turbine as a heatmap in wind speed.

    df_offsets should be a dataframe with columns:
       - wind_direction,
       - wind_speed,
       - turbine identifiers (possibly multiple)

    Produces a heat map of the offsets for all wind directions and
    wind speeds for turbine specified by turb_id. Dataframe is assumed
    to contain individual turbine offsets in distinct columns (unlike
    the yaw_angles_opt column from FLORIS.

    Args:
        df_offsets (pd.DataFrame): dataframe with offsets
        turb_id (int or str): turbine id or column name
        ax (matplotlib.axes.Axes): axis to plot on.  If None, a new figure is created.
            Default is None.

    Returns:
        A tuple containing a matplotlib.axes.Axes object and a matplotlib.colorbar.Colorbar

    """
    if isinstance(turb_id, int):
        if "yaw_angles_opt" in df_offsets.columns:
            offsets = np.vstack(df_offsets.yaw_angles_opt.to_numpy())[:, turb_id]
            df_offsets = pd.DataFrame(
                {
                    "wind_direction": df_offsets.wind_direction,
                    "wind_speed": df_offsets.wind_speed,
                    "yaw_offset": offsets,
                }
            )
            turb_id = "yaw_offset"
        else:
            raise TypeError(
                "Specify turb_id as a full string for the " + "correct dataframe column."
            )

    ws_array = np.unique(df_offsets.wind_speed)
    wd_array = np.unique(df_offsets.wind_direction)

    # Construct array of offets
    offsets_array = np.zeros((len(ws_array), len(wd_array)))
    for i, ws in enumerate(ws_array):
        offsets_array[-i, :] = df_offsets[df_offsets.wind_speed == ws][turb_id].values

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    d_wd = (wd_array[1] - wd_array[0]) / 2
    d_ws = (ws_array[1] - ws_array[0]) / 2
    im = ax.imshow(
        offsets_array,
        interpolation=None,
        extent=[wd_array[0] - d_wd, wd_array[-1] + d_wd, ws_array[0] - d_ws, ws_array[-1] + d_ws],
        aspect="auto",
    )
    ax.set_xlabel("Wind direction")
    ax.set_ylabel("Wind speed")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("Yaw offset")

    return ax, cbar


# TODO: This function feels a little old fashioned
def plot_offsets_wd(df_offsets, turb_id, ws_plot, color="black", alpha=1.0, label=None, ax=None):
    """Plot offsets for a single turbine as a function of wind direction.

    df_offsets should be a dataframe with columns:
       - wind_direction,
       - wind_speed,
       - turbine identifiers (possibly multiple)

    if ws_plot is scalar, only that wind speed is plotted. If ws_plot is
    a two-element tuple or list, that range of wind speeds is plotted.

    label only allowed is single wind speed is given.

    Args:
        df_offsets (pd.DataFrame): dataframe with offsets
        turb_id (int or str): turbine id or column name
        ws_plot (float or list): wind speed to plot
        color (str): color of line
        alpha (float): transparency of line
        label (str): label for line
        ax (matplotlib.axes.Axes): axis to plot on.  If None, a new figure is created.
            Default is None.
    """
    if isinstance(turb_id, int):
        if "yaw_angles_opt" in df_offsets.columns:
            offsets = np.vstack(df_offsets.yaw_angles_opt.to_numpy())[:, turb_id]
            df_offsets = pd.DataFrame(
                {
                    "wind_direction": df_offsets.wind_direction,
                    "wind_speed": df_offsets.wind_speed,
                    "yaw_offset": offsets,
                }
            )
            turb_id = "yaw_offset"
        else:
            raise TypeError(
                "Specify turb_id as a full string for the " + "correct dataframe column."
            )

    if hasattr(ws_plot, "__len__") and label is not None:
        label = None
        logger.warn("label option can only be used for single wind speed plot.")

    ws_array = np.unique(df_offsets.wind_speed)
    wd_array = np.unique(df_offsets.wind_direction)

    if hasattr(ws_plot, "__len__"):
        offsets_list = []
        for ws in ws_array:
            if ws >= ws_plot[0] and ws <= ws_plot[-1]:
                offsets_list.append(df_offsets[df_offsets.wind_speed == ws][turb_id].values)
    else:
        offsets_list = [df_offsets[df_offsets.wind_speed == ws_plot][turb_id].values]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for offsets in offsets_list:
        ax.plot(wd_array, offsets, color=color, alpha=alpha, label=label)

    ax.set_xlabel("Wind direction")
    ax.set_ylabel("Yaw offset")

    return ax
