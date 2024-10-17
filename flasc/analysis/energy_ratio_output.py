"""Store the results of the energy ratio calculations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import matplotlib.axes._axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flasc.analysis.energy_ratio_input import EnergyRatioInput
from flasc.utilities.energy_ratio_utilities import (
    add_wd_bin,
    add_ws_bin,
    filter_all_nulls,
    filter_any_nulls,
)


class EnergyRatioOutput:
    """Store the results of the energy ratio calculations.

    Additionally provide convenient methods for plotting and saving the results.
    """

    def __init__(
        self,
        df_result: pd.DataFrame,
        er_in: EnergyRatioInput,
        df_freq: pd.DataFrame,
        ref_cols: List[str],
        test_cols: List[str],
        wd_cols: List[str],
        ws_cols: List[str],
        uplift_cols: List[str],
        wd_step: float,
        wd_min: float,
        wd_max: float,
        ws_step: float,
        ws_min: float,
        ws_max: float,
        bin_cols_in: List[str],
        weight_by: str,
        wd_bin_overlap_radius: float,
        N: int,
        remove_all_nulls: bool = False,
    ) -> None:
        """Initialize an EnergyRatioOutput object.

        Args:
            df_result (pd.DataFrame): The energy ratio results.
            er_in (EnergyRatioInput): The energy table used in the energy ratio calculation.
            df_freq (pd.DataFrame): Weights used for bins.
            ref_cols (List[str]): The column names of the reference turbines.
            test_cols (List[str]): The column names of the test wind turbines.
            wd_cols (List[str]): The column names of the wind directions.
            ws_cols (List[str]): The column names of the wind speeds.
            uplift_cols (List[str]): The column names of the uplifts.
            wd_step (float): The wind direction bin size.
            wd_min (float): The minimum wind direction value.
            wd_max (float): The maximum wind direction value.
            ws_step (float): The wind speed bin size.
            ws_min (float): The minimum wind speed value.
            ws_max (float): The maximum wind speed value.
            bin_cols_in (List[str]): TBD
            weight_by (str): How to weight the energy ratio, options are 'min', or 'sum'.
                'min' means the minimum count across the dataframes
                is used to weight the energy ratio.
                'sum' means the sum of the counts
                across the dataframes is used to weight the energy ratio.
            wd_bin_overlap_radius (float): The radius of overlap between wind direction bins.
            N (int): The number of bootstrap iterations used in the energy ratio calculation.
            remove_all_nulls: (bool): Construct reference and test by
                strictly requiring all data to be
                available. If False, a minimum one data point from
                ref_cols, test_cols, wd_cols, and ws_cols
                must be available to compute the bin. Defaults to False.
        """
        self.df_result = df_result
        self.df_freq = df_freq
        self.df_names = er_in.df_names
        self.num_df = len(self.df_names)
        self.er_in = er_in
        self.ref_cols = ref_cols
        self.test_cols = test_cols
        self.wd_cols = wd_cols
        self.ws_cols = ws_cols
        self.uplift_cols = uplift_cols
        self.wd_step = wd_step
        self.wd_min = wd_min
        self.wd_max = wd_max
        self.ws_step = ws_step
        self.ws_min = ws_min
        self.ws_max = ws_max
        self.bin_cols_in = bin_cols_in
        self.weight_by = weight_by
        self.wd_bin_overlap_radius = wd_bin_overlap_radius
        self.N = N
        self.remove_all_nulls = remove_all_nulls

    def plot_energy_ratios(
        self,
        df_names_subset: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        color_dict: Optional[Dict[str, Any]] = None,
        axarr: Optional[Union[axes.Axes, List[axes.Axes]]] = None,
        polar_plot: bool = False,
        show_wind_direction_distribution: bool = True,
        show_wind_speed_distribution: bool | None = None,
        overlay_frequency: bool = False,
        _is_uplift: bool = False,
    ) -> Union[axes.Axes, List[axes.Axes]]:
        """Plot the energy ratios.

        Args:
            df_names_subset (Optional[List[str]], optional): A subset of the dataframes
                used in the energy ratio calculation. Defaults to None.
            labels (Optional[List[str]], optional): The labels for the energy ratios.
                 Defaults to None.
            color_dict (Optional[Dict[str, Any]], optional): A dictionary
                mapping labels to colors. Defaults to None.
            axarr (Optional[Union[axes.Axes, List[axes.Axes]]], optional): The axes
                to plot on. Defaults to None.
            polar_plot (bool, optional): Whether to plot the energy ratios
                on a polar plot. Defaults to False.
            show_wind_direction_distribution (bool, optional): Whether to show
                 the wind direction distribution. Defaults to True.
            show_wind_speed_distribution (bool, optional): Whether to show
                the wind speed distribution. Defaults to True, unless polar_plot is True.
            overlay_frequency (bool, optional): Whether to plot the
                frequency distribution used for calculation.
            _is_uplift (bool, optional): Whether being called by plot_uplift(). Defaults to False.

        Returns:
            Union[axes.Axes, List[axes.Axes]]: The axes used for plotting.

        Raises:
            ValueError: If show_wind_speed_distribution is True and polar_plot is True.

        Notes:
            - If df_names_subset is None, all dataframes will be plotted.
            - If df_names_subset is not a list, it will be converted to a list.
            - If labels is None, the dataframe names will be used as labels.
            - If color_dict is None, a default color scheme will be used.
            - If axarr is None, a new figure will be created.
        """
        # Handle defaults for show_wind_speed_distribution
        if show_wind_speed_distribution is None:
            if polar_plot:
                show_wind_speed_distribution = False
            else:
                show_wind_speed_distribution = True

        # Only allow showing the wind speed distribution if polar_plot is False
        if polar_plot and show_wind_speed_distribution:
            raise ValueError("show_wind_speed_distribution cannot be True if polar_plot is True")

        # If df_names_subset is None, plot all the dataframes
        if df_names_subset is None:
            df_names_subset = self.df_names

        # If df_names_subset is not a list, convert it to a list
        if not isinstance(df_names_subset, list):
            df_names_subset = [df_names_subset]

        # Total number of energy ratios to plot
        N = len(df_names_subset)

        # If labels is None, use the dataframe names
        if labels is None:
            labels = df_names_subset

        # If labels is not a list, convert it to a list
        if not isinstance(labels, list):
            labels = [labels]

        # Confirm that the length of labels is the same as the length of df_names_subset
        if len(labels) != N:
            raise ValueError("Length of labels must be the same as the length of df_names_subset")

        # Generate the default colors using the seaborn color palette
        default_colors = sns.color_palette("colorblind", N)

        # If color_dict is None, use the default colors
        if color_dict is None:
            color_dict = {labels[i]: default_colors[i] for i in range(N)}
            color_dict["weight"] = "black"

        # If color_dict is not a dictionary, raise an error
        if not isinstance(color_dict, dict):
            raise ValueError("color_dict must be a dictionary")

        # Make sure the keys of color_dict are in labels
        if not all([label in labels + ["weight"] for label in color_dict.keys()]):
            raise ValueError("color_dict keys must be in df_names_subset")

        if axarr is None:
            if polar_plot:
                _, axarr = plt.subplots(
                    nrows=1, ncols=2, figsize=(10, 5), subplot_kw={"projection": "polar"}
                )
            else:
                if show_wind_direction_distribution:
                    if show_wind_speed_distribution:
                        num_rows = 3  # Add rows to show wind speed and wind direction distribution
                    else:
                        num_rows = 2  # Add rows to show wind direction distribution
                else:
                    num_rows = 1
                _, axarr = plt.subplots(
                    nrows=num_rows, ncols=1, sharex=True, figsize=(11, num_rows * 3)
                )
        else:  # Confirm correct number of axes passed in
            if polar_plot:
                if len(axarr) != 2:
                    raise ValueError("If polar_plot is True, axarr must have length of 2")
            else:
                if show_wind_direction_distribution:
                    if show_wind_speed_distribution:
                        if len(axarr) != 3:
                            raise ValueError(
                                "If show_wind_speed_distribution and "
                                "show_wind_direction_distribution are"
                                " True, axarr must have length of 3"
                            )
                    else:
                        if len(axarr) != 2:
                            raise ValueError(
                                "If show_wind_direction_distribution is True, and"
                                " show_wind_direction"
                                " is False axarr must have length of 2"
                            )
                else:
                    # Confirm axarr is of type Axes
                    if not isinstance(axarr, plt.Axes):
                        raise ValueError(
                            "If show_wind_direction_distribution and show_wind_speed_distribution"
                            " are False, axarr be of type matplotlib.pyplot.Axes "
                            "and not a list of axes"
                        )

        # For plotting, create a copy in case
        df = self.df_result.copy()

        # Get x-axis values
        x = np.array(df["wd_bin"], dtype=float)

        # Get xlims to add a horizontal line at 1
        xlims = np.linspace(np.min(x) - 4.0, np.max(x) + 4.0, 1000)

        if polar_plot:
            x = (90.0 - x) * np.pi / 180.0  # Convert to radians
            xlims = (90.0 - xlims) * np.pi / 180.0  # Convert to radians

        # Add NaNs to avoid connecting plots over gaps
        dwd = np.min(x[1::] - x[0:-1])
        jumps = np.where(np.diff(x) > dwd * 1.50)[0]
        if len(jumps) > 0:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "wd_bin": x[jumps] + dwd / 2.0,
                            "N_bin": [0] * len(jumps),
                        }
                    ),
                ],
                axis=0,
                ignore_index=False,
            )
            df = df.iloc[np.argsort(df["wd_bin"])].reset_index(drop=True)
            x = np.array(df["wd_bin"], dtype=float)
            if polar_plot:  # Convert to radians
                x = (90.0 - x) * np.pi / 180.0

        # Energy ratio plot ========================================
        if show_wind_direction_distribution:
            ax = axarr[0]
        else:
            ax = axarr

        # Plot the horizontal line at 1 or 0
        if _is_uplift:
            ax.plot(xlims, np.zeros_like(xlims), color="black")
        else:
            ax.plot(xlims, np.ones_like(xlims), color="black")

        # Plot the energy ratios
        for df_name, label in zip(df_names_subset, labels):
            ax.plot(x, df[df_name], "-o", markersize=3.0, label=label, color=color_dict[label])

            # If data includes upper and lower bounds plot them
            if df_name + "_ub" in df.columns:
                ax.fill_between(
                    x,
                    df[df_name + "_lb"],
                    df[df_name + "_ub"],
                    alpha=0.25,
                    color=color_dict[label],
                )

        # Format the energy ratio plot
        ax.legend()
        ax.grid(visible=True, which="major", axis="both", color="gray")
        ax.grid(visible=True, which="minor", axis="both", color="lightgray")
        ax.minorticks_on()
        ax.set_title("Energy Ratio")
        ax.set_ylabel("Energy Ratio")

        # Wind Direction Bin Plot ========================================
        if not show_wind_direction_distribution:
            ax.set_xlabel("Wind Direction (deg)")
            return axarr

        ax = axarr[1]

        # Set the bar width using self.wd_step
        bar_width = (0.7 / N) * self.wd_step
        if polar_plot:
            bar_width = bar_width * np.pi / 180.0

        for i, (df_name, label) in enumerate(zip(df_names_subset, labels)):
            if _is_uplift:  # Special case, use the minimum or the sum
                ax.set_title(
                    "Minimum of Points per Bin"
                    if self.weight_by == "min"
                    else "Sum of Points per Bin"
                )
            else:
                ax.set_title("Number of Points per Bin")

            x = np.array(self.df_result["wd_bin"], dtype=float)
            if polar_plot:  # Convert to radians
                x = (90.0 - x) * np.pi / 180.0
            ax.bar(
                x - (i - N / 2) * bar_width,
                self.df_result["count_" + df_name],
                width=bar_width,
                label=label,
                color=color_dict[label],
            )
        if overlay_frequency:
            if "weight" in color_dict:
                col = color_dict["weight"]
            else:
                col = "black"
            df_wd_weight = self.df_freq.drop(columns="ws_bin").groupby("wd_bin").sum().reset_index()
            ax.plot(df_wd_weight["wd_bin"], df_wd_weight["weight"], color=col, label="Weight")

        ax.legend()
        ax.set_ylabel("Number of Points")

        ax.grid(True)

        # Wind Speed Distribtution Plot ========================================
        if not show_wind_speed_distribution:
            ax.set_xlabel("Wind Direction (deg)")
            return axarr

        ax = axarr[2]

        df_bin_counts = self._compute_ws_counts()
        sns.scatterplot(
            data=df_bin_counts,
            x="wd_bin",
            y="ws_bin",
            size="count",
            hue="count",
            ax=ax,
            legend=True,
            color="k",
        )
        ax.set_title(
            "Minimum Number of Points per Bin"
            if self.weight_by == "min"
            else "Sum of Points per Bin"
        )
        ax.set_xlabel("Wind Direction (deg)")
        ax.set_ylabel("Wind Speed (m/s)")

        ax.grid(True)

        return axarr

    def plot_uplift(
        self,
        uplift_names_subset: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        color_dict: Optional[Dict[str, Any]] = None,
        axarr: Optional[Union[axes.Axes, List[axes.Axes]]] = None,
        polar_plot: bool = False,
        show_wind_direction_distribution: bool = True,
        show_wind_speed_distribution: bool = True,
        overlay_frequency: bool = False,
    ) -> Union[axes.Axes, List[axes.Axes]]:
        """Plot the uplift in energy ratio.

        Args:
            uplift_names_subset (Optional[List[str]], optional): A subset
                of the uplifts computed to print. Defaults to None.
            labels (Optional[List[str]], optional): The labels for the uplifts. Defaults to None.
            color_dict (Optional[Dict[str, Any]], optional): A dictionary
                mapping labels to colors. Defaults to None.
            axarr (Optional[Union[axes.Axes, List[axes.Axes]]], optional): The axes
                to plot on. Defaults to None.
            polar_plot (bool, optional): Whether to plot the uplift on
                a polar plot. Defaults to False.
            show_wind_direction_distribution (bool, optional): Whether to
                show the wind direction distribution. Defaults to True.
            show_wind_speed_distribution (bool, optional): Whether to
                show the wind speed distribution. Defaults to True, unless polar_plot is True.
            overlay_frequency (bool, optional): Whether to
                plot the frequency distribution used for calculation.

        Raises:
            ValueError: If show_wind_speed_distribution is True and polar_plot is True.

        Returns:
            Union[axes.Axes, List[axes.Axes]]: The axes used for plotting.

        Notes:
            - If axarr is None, a new figure will be created.
            - If axarr is a single axes object, it will be used to plot the uplift.
            - If axarr is a list of axes objects, each
                component of the uplift will be plotted on a separate axes object.
            - If polar_plot is True, the uplift will be plotted on a polar plot.
            - If show_wind_direction_distribution is True,
                the wind direction distribution will be shown.
            - If show_wind_speed_distribution is True, the wind speed distribution will be shown.
        """
        # Handle defaults for show_wind_speed_distribution
        if show_wind_direction_distribution is None:
            if polar_plot:
                show_wind_direction_distribution = False
            else:
                show_wind_direction_distribution = True

        # Only allow showing the wind speed distribution if polar_plot is False
        if polar_plot and show_wind_speed_distribution:
            raise ValueError("show_wind_speed_distribution cannot be True if polar_plot is True")

        # If df_names_subset is None, plot all the dataframes
        if uplift_names_subset is None:
            uplift_names_subset = self.uplift_cols

        # If df_names_subset is not a list, convert it to a list
        if not isinstance(uplift_names_subset, list):
            uplift_names_subset = [uplift_names_subset]

        # Total number of energy ratios to plot
        N = len(uplift_names_subset)

        if N == 0:
            raise ValueError(
                "No uplifts to plot. Please specify uplifts when calling compute_energy_ratio()."
            )

        # If labels is None, use the dataframe names
        if labels is None:
            labels = uplift_names_subset

        # If labels is not a list, convert it to a list
        if not isinstance(labels, list):
            labels = [labels]

        # Confirm that the length of labels is the same as the length of df_names_subset
        if len(labels) != N:
            raise ValueError(
                "Length of labels must be the same as the length of uplift_names_subset"
            )

        # Generate the default colors using the seaborn color palette
        default_colors = sns.color_palette("colorblind", N)

        # If color_dict is None, use the default colors
        if color_dict is None:
            color_dict = {labels[i]: default_colors[i] for i in range(N)}
            color_dict["weight"] = "black"

        # If color_dict is not a dictionary, raise an error
        if not isinstance(color_dict, dict):
            raise ValueError("color_dict must be a dictionary")

        # Make sure the keys of color_dict are in labels
        if not all([label in labels + ["weight"] for label in color_dict.keys()]):
            raise ValueError("color_dict keys must be in df_names_subset")

        if axarr is None:
            if polar_plot:
                _, axarr = plt.subplots(
                    nrows=1, ncols=2, figsize=(10, 5), subplot_kw={"projection": "polar"}
                )
            else:
                if show_wind_direction_distribution:
                    if show_wind_speed_distribution:
                        num_rows = 3  # Add rows to show wind speed and wind direction distribution
                    else:
                        num_rows = 2  # Add rows to show wind direction distribution
                else:
                    num_rows = 1
                _, axarr = plt.subplots(
                    nrows=num_rows, ncols=1, sharex=True, figsize=(11, num_rows * 3)
                )

        self.plot_energy_ratios(
            df_names_subset=uplift_names_subset,
            labels=labels,
            color_dict=color_dict,
            axarr=axarr,
            polar_plot=polar_plot,
            show_wind_direction_distribution=show_wind_direction_distribution,
            show_wind_speed_distribution=show_wind_speed_distribution,
            overlay_frequency=overlay_frequency,
            _is_uplift=True,
        )

        # Finish plots
        if not show_wind_direction_distribution:
            ax = axarr
        else:
            ax = axarr[0]

        # Finish Energy Ratio plot
        ax.set_ylabel("Percent Change")
        ax.set_title("Uplift in Energy Ratio")

        if not show_wind_direction_distribution:
            return axarr

        # # Finish Wind Direction Distribution plot
        # ax = axarr[1]
        # ax.set_title("Minimum Number of Points per Bin")

        return axarr

    def _compute_ws_counts(self):
        """Compute the ws bin counts."""
        # Temporary copy of energy table
        df_ = self.er_in.get_df()

        # Filter df_ to remove null values
        null_filter = filter_all_nulls if self.remove_all_nulls else filter_any_nulls
        df_ = null_filter(df_, self.ref_cols, self.test_cols, self.ws_cols, self.wd_cols)

        # Assign the wd/ws bins
        df_ = add_ws_bin(
            df_,
            self.ws_cols,
            self.ws_step,
            self.ws_min,
            self.ws_max,
            remove_all_nulls=self.remove_all_nulls,
        )
        df_ = add_wd_bin(
            df_,
            self.wd_cols,
            self.wd_step,
            self.wd_min,
            self.wd_max,
            remove_all_nulls=self.remove_all_nulls,
        )

        # Get the bin count by wd, ws and df_name
        df_group = df_.group_by(["wd_bin", "ws_bin", "df_name"]).count()

        # Collect the minimum number of points per bin
        df_return = (
            df_group.group_by(["wd_bin", "ws_bin"]).min()
            if self.weight_by == "min"
            else df_group.group_by(["wd_bin", "ws_bin"]).sum()
        )

        return df_return.drop("df_name").to_pandas()
