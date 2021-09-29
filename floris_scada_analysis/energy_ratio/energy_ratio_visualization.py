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

import matplotlib.pyplot as plt


def plot(energy_ratios, labels=None):
    # Format inputs if single case is inserted vs. lists
    if not isinstance(energy_ratios, (list, tuple)):
        energy_ratios = [energy_ratios]
        if isinstance(labels, str):
            labels = [labels]

    if labels is None:
        labels = ["Nominal" for _ in energy_ratios]
        uq_labels = ["Confidence bounds" for _ in energy_ratios]
    else:
        uq_labels = ["%s confidence bounds" % lb for lb in labels]

    N = len(energy_ratios)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))

    # Calculate bar width for bin counts
    bar_width = (0.7 / N) * np.min(
        [np.diff(er["wd_bin"])[0] for er in energy_ratios]
    )

    for ii, df in enumerate(energy_ratios):
        df = df.copy()

        # Get x-axis values
        x = np.array(df["wd_bin"], dtype=float)

        # Add NaNs to avoid connecting plots over gaps
        dwd = np.min(x[1::] - x[0:-1])
        jumps = np.where(np.diff(x) > dwd * 1.50)[0]
        if len(jumps) > 0:
            df = df.append(
                pd.DataFrame(
                    {
                        'wd_bin': x[jumps] + dwd / 2.0,
                        "N_bin": [0] * len(jumps),
                    }
                )
            )
            df = df.iloc[np.argsort(df['wd_bin'])].reset_index(drop=True)
            x = np.array(df["wd_bin"], dtype=float)

        # Plot horizontal black line at 1.
        xlims = [np.min(x) - 4., np.max(x) + 4.]
        ax[0].plot(xlims, [1., 1.], color='black')

        # Plot energy ratios
        ax[0].plot(x, df["baseline"], '-o', markersize=3., label=labels[ii])

        # Plot uncertainty bounds from bootstrapping, if applicable
        has_uq = (np.max(np.abs(df['baseline'] - df['baseline_l'])) > 0.001)
        if has_uq:
            ax[0].fill_between(x, df["baseline_l"], df["baseline_u"],
                               alpha=0.25, label=uq_labels[ii])

        # Plot the bin count
        ax[1].bar(x-(ii-N/2)*bar_width, df["N_bin"], width=bar_width)

    # Format the energy ratio plot
    ax[0].set_ylabel('Energy ratio (-)')
    ax[0].legend()
    ax[0].grid(b=True, which='major', axis='both', color='gray')
    ax[0].grid(b=True, which='minor', axis='both', color='lightgray')
    ax[0].minorticks_on()
    plt.grid(True)

    if labels[0] is not None:
        ax[0].legend()

    # Format the bin count plot
    ax[1].grid(b=True, which='major', axis='both', color='gray')
    ax[1].grid(b=True, which='minor', axis='both', color='lightgray')
    ax[1].set_xlabel('Wind direction (deg)')
    ax[1].set_ylabel('Number of data points (-)')

    # Enforce a tight layout
    plt.tight_layout()

    return fig, ax
