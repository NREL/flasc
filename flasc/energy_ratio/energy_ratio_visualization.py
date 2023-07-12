# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def plot(
    energy_ratios,
    df_freqs=None,
    labels=None, 
    colors=None, 
    hide_uq_labels=True,
    polar_plot=False,
    axarr=None,
    show_barplot_legend=True,
):
    """This function plots energy ratios against the reference wind
    direction. The plot may or may not include uncertainty bounds,
    depending on the information contained in the provided energy ratio
    dataframes.

    Args:
        energy_ratios ([iteratible]): List of Pandas DataFrames containing
                the energy ratios for each dataset, respectively. Each
                entry in this list is a Dataframe containing the found
                energy ratios under the prespecified settings, contains the
                columns:
                    * wd_bin: The mean wind direction for this bin
                    * N_bin: Number of data entries in this bin
                    * baseline: Nominal energy ratio value (without UQ)
                    * baseline_l: Lower bound for energy ratio. This
                        value is equal to baseline without UQ and lower
                        with UQ.
                    * baseline_u: Upper bound for energy ratio. This
                        value is equal to baseline without UQ and higher
                        with UQ.
        df_freqs ([iteratible], optional): List of Pandas DataFrames containing
            the wind rose distributions for each dataset that is plotted. These
            dataframes contain, at the minimum, the columns:
                    * wd_bin_edges: a Pandas Interval specifying the bounds
                         of the wind direction bin.
                    * ws_bin_edges: a Pandas Interval specifying the bounds
                         of the wind speed bin.
                    * freq: a float specifying the frequency of occurrence
                         of this particular wind speed and wind direction bin.
        labels ([iteratible], optional): Label for each of the energy ratio
            dataframes. Defaults to None.
        colors ([iteratible], optional): Colors for the energy ratio plots,
            being a list with one color for each dataset plotted. Thus,
            colors is a list-like object with number of entries equal
            to the length of the list 'energy_ratios'. If None, will
            default to the matplotlib.pyplot tableau color palette.
        hide_uq_labels (bool, optional): If true, do not specifically label
            the confidence intervals in the plot
        polar_plot (bool, optional): Plots the energy ratios in a polar
           coordinate system, aligned with the wind direction coordinate
           system of FLORIS. Defaults to False.
        show_barplot_legend (bool, optional): Show the legend in the bar
            plot figure?  Defaults to True

    Returns:
        axarr([iteratible]): List of axes in the figure with length 2.
    """
    # Format inputs if single case is inserted vs. lists
    if not isinstance(energy_ratios, (list, tuple)):
        energy_ratios = [energy_ratios]
    if isinstance(labels, str):
        labels = [labels]

    # Format df_freqs
    if df_freqs is not None:
        if not isinstance(df_freqs, (list, tuple)):
            df_freqs = [df_freqs]

    if labels is None:
        labels = ["Nominal" for _ in energy_ratios]
        uq_labels = ["Confidence bounds" for _ in energy_ratios]
    else:
        uq_labels = ["%s confidence bounds" % lb for lb in labels]

    if hide_uq_labels:
        uq_labels = ['_nolegend_' for l in uq_labels]

    # Come up with hatch patterns
    hatch_patterns = [
        '//', '++', 'xx', 'oo', '\\\\', '--', 'OO', '..', '**', '||',
        '/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-',
        '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*',
    ] * 10  # Repeat 10 times to ensure always sufficiently large size

    # If colors is a list that contains None, assume colors not assigned
    if isinstance(colors, (list, tuple)):
        if None in colors:
            if not all(v is None for v in colors):
                print("It's possible some but not all colors are supplied, therefore reverting to defaults")
            colors = None

    if colors is None:
        # If nothing specified, use the default tableau colors from matplotlib.pyplot
        colors = [mcolors.to_rgb(clr) for clr in mcolors.TABLEAU_COLORS.keys()]

    # In case colors is specified by a character, convert to RGB
    colors = [mcolors.to_rgb(clr) for clr in colors]


    N = len(energy_ratios)
    if axarr is None:
        if polar_plot:
            _, axarr = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), subplot_kw={'projection': 'polar'})
        else:
            _, axarr = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 5))

    # Calculate bar width for bin counts
    bar_width = (0.7 / N) * np.min(
        [er["wd_bin"].diff().min() for er in energy_ratios]
    )
    if polar_plot:
        bar_width = bar_width * np.pi / 180.0

    for ii, df in enumerate(energy_ratios):
        df = df.copy()

        if df.shape[0] < 2:
            # Do not plot single values
            continue

        # Get x-axis values
        x = np.array(df["wd_bin"], dtype=float)
    
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
                    )
                ],
                axis=0,
                ignore_index=False,
            )
            df = df.iloc[np.argsort(df["wd_bin"])].reset_index(drop=True)
            x = np.array(df["wd_bin"], dtype=float)

        # Plot horizontal black line at 1.
        xlims = np.linspace(np.min(x) - 4.0, np.max(x) + 4.0, 1000)

        if polar_plot:
            x = (90.0 - x) * np.pi / 180.0  # Convert to radians
            xlims = (90.0 - xlims) * np.pi / 180.0  # Convert to radians

        # Plot energy ratios
        axarr[0].plot(xlims, np.ones_like(xlims), color="black")
        axarr[0].plot(x, df["baseline"], "-o", markersize=3.0, label=labels[ii], color=colors[ii])

        # Plot uncertainty bounds from bootstrapping, if applicable
        has_uq = np.max(np.abs(df["baseline"] - df["baseline_lb"])) > 0.001
        if has_uq:
            axarr[0].fill_between(
                x,
                df["baseline_lb"],
                df["baseline_ub"],
                alpha=0.25,
                label=uq_labels[ii],
                color=colors[ii]
            )

    # Plot the bin count
    is_none = False
    if df_freqs is None:
        is_none = True
    elif isinstance(df_freqs, list):
        is_none = np.any([c is None for c in df_freqs])

    if not is_none:
        for ii, df_freq in enumerate(df_freqs):
            wd_bins = df_freq["wd_bin"].unique()
            n_ws_bins = len(df_freq["ws_bin_edges"].unique())
            alpha_array = np.linspace(0.0, 0.8, n_ws_bins)  # Bar colors are a transient from full color to mix of 20% color and 80% white
            # # Actual plots

            x = wd_bins
            if N > 1:
                x = x + (ii - N / 2) * bar_width

            bottom = np.zeros(len(wd_bins), dtype=float)
            for wsii, ws_bin_edges in enumerate(np.sort(df_freq["ws_bin_edges"].unique())):
                # Mix nominal bar with white to mimic alpha but have no background transparency
                bar_color = colors[ii] + alpha_array[wsii] * (1.0 - np.array(colors[ii]))
                bin_info = df_freq[df_freq["ws_bin_edges"] == ws_bin_edges]
                bin_map = [np.where(wd == wd_bins)[0][0] for wd in bin_info["wd_bin"]]
                y = np.zeros_like(wd_bins)
                y[bin_map] = np.array(bin_info["freq"], dtype=float)

                if polar_plot:
                    x = (90.0 - wd_bins) * np.pi / 180.0

                # Plot the bar on top of existing bar
                axarr[1].bar(
                    x=x,
                    height=y,
                    width=bar_width,
                    bottom=bottom,
                    label="{:s}: {}".format(labels[ii], str(ws_bin_edges)),
                    hatch=hatch_patterns[wsii],
                    color=bar_color,
                    edgecolor=colors[ii],
                    linewidth=0.5,
                )

                # Increment bar heights
                bottom = bottom + y
    else:
        axarr[1].bar(x - (ii - N / 2) * bar_width, df["bin_count"], width=bar_width, color=colors[ii])

    # Format the energy ratio plot
    axarr[0].legend()
    axarr[0].grid(visible=True, which="major", axis="both", color="gray")
    axarr[0].grid(visible=True, which="minor", axis="both", color="lightgray")
    axarr[0].minorticks_on()
    plt.grid(True)

    if labels[0] is not None:
        axarr[0].legend()

    # Format the bin count plot
    axarr[1].grid(visible=True, which="major", axis="both", color="gray")
    axarr[1].grid(visible=True, which="minor", axis="both", color="lightgray")
    if show_barplot_legend:
        if df_freqs is not None:
            axarr[1].legend(ncols=len(df_freqs), fontsize="small")

    # Arrange xtick labels to align with FLORIS internal coordinate system
    if polar_plot:
        axarr[0].set_title("Energy ratio (-)")
        axarr[1].set_title("Number of data points (-)")
        for axx in axarr:
            xticks = np.remainder(90.0 - axx.get_xticks() * 180.0 / np.pi + 360.0, 360.0)
            axx.set_xticklabels(["{:.0f}°".format(x) for x in xticks])
    else:
        axarr[0].set_ylabel("Energy ratio (-)")
        axarr[1].set_xlabel("Wind direction (deg)")
        axarr[1].set_ylabel("Number of data points (-)")

    # Enforce a tight layout
    plt.tight_layout()
    return axarr

