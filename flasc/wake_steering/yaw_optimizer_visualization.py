# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_uplifts_by_atmospheric_conditions(
    df_list,
    labels=None,
    ws_edges=np.arange(3.0, 17.0, 1.0),
    wd_edges=np.arange(0.0, 360.0001, 3.0),
    ti_edges = np.arange(0.0, 0.30, 0.02),
):
    # Calculate bin means
    ws_labels = (ws_edges[0:-1] + ws_edges[1::]) / 2.
    wd_labels = (wd_edges[0:-1] + wd_edges[1::]) / 2.
    ti_labels = (ti_edges[0:-1] + ti_edges[1::]) / 2.

    # Format input
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]

    # Clean dataframes and calculate AEP gains
    for ii, df in enumerate(df_list):
         # Only keep cases with Pbl > 0 and non-NaNs
        df = df.dropna(how='any', subset=['farm_power_baseline', 'farm_power_opt'])
        df = df[df['farm_power_baseline'] > 0.01].reset_index(drop=True)

        # Check if frequency vector exists
        if not "farm_energy_baseline" in df.columns:
            if "frequency" in df.columns:
                df["frequency"] = df["frequency"].astype(float)
            elif "freq" in df.columns:
                df["frequency"] = df["freq"].astype(float)
            else:
                df["frequency"] = 1.0
                print(
                    "No column 'freq' or 'frequency' found in dataframe." +
                    "Assuming a uniform distribution."
                )

            # Calculate wind farm energy, baseline and optimized
            df["farm_energy_baseline"] = df["farm_power_baseline"] * df["frequency"]
            df["farm_energy_opt"] = df["farm_power_opt"] * df["frequency"]

        # Calculate relative and absolute uplift in energy for every condition
        df["Prel"] = np.where(
            df["farm_power_baseline"] > 0.0,
            df["farm_power_opt"] / df["farm_power_baseline"],
            0.0
        )

        df["Pabs"] = (
            (df["farm_energy_opt"] - df["farm_energy_baseline"]) /
            np.nansum(df["farm_energy_opt"] - df["farm_energy_baseline"])
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
        ax[1].set_ylabel('Frequency (-)')
        ax[1].set_xlabel(xlabel)
        ax[1].grid(True)
        ax[1].set_xticks(x[dii])
        if len(x[dii]) > 50:  # Too many ticks: reduce
            xtlabels = ['' for _ in range(len(x[dii]))]
            xtlabels[0::5] = ['%.1f' % i for i in x[dii][0::5]]
        else:
            xtlabels = ['%.1f' % i for i in x[dii]]
        ax[1].set_xticklabels(xtlabels)

    if not np.all([a is None for a in labels]):
        ax[0].legend()
        ax[1].legend()

    return fig, ax
