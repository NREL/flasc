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

import matplotlib.pyplot as plt


def plot(energy_ratios, labels=None):
    # Format inputs if single case is inserted vs. lists
    if not isinstance(energy_ratios, (list, tuple)):
        energy_ratios = [energy_ratios]
        if isinstance(labels, str):
            labels = [labels]

    if labels is None:
        labels = [None for _ in energy_ratios]

    N = len(energy_ratios)
    fig, ax = plt.subplots(nrows=N, sharex=True)

    # Calculate bar width for bin counts
    bar_width = (0.7 / N) * np.min(
        [np.diff(er["wd_bin"])[0] for er in energy_ratios]
    )

    for ii, df in enumerate(energy_ratios):
        # Get x-axis values
        x = np.array(df["wd_bin"], dtype=float)

        # Plot horizontal black line at 1.
        xlims = [np.min(x) - 4., np.max(x) + 4.]
        ax[0].plot(xlims, [1., 1.], color='black')

        # Plot energy ratios
        ax[0].plot(x, df["baseline"], '-o', label=labels[ii])

        # Plot uncertainty bounds from bootstrapping
        ax[0].fill_between(x, df["baseline_l"], df["baseline_u"], alpha=0.15)

        # Plot the bin count
        ax[1].bar(x-(ii-N/2)*bar_width, df["N_bin"], width=bar_width)

    # Format the energy ratio plot
    ax[0].set_ylabel('Energy ratio (-)')
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
