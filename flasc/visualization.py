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
from matplotlib import pyplot as plt


def plot_with_wrapping(x, y, ax=None, high=360.0, linestyle="-", marker=None, color="black", label=None):
    # Create figure, if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Format inputs to numpy arrays
    x = np.array(x, copy=True)
    y = np.array(y, copy=True, dtype=float)
    y = np.remainder(y, high)

    # Initialize empty arrays
    xw = np.array(x, copy=True)[0:0]
    yw = np.array(y, copy=True)[0:0]

    # Deal with wrapping
    id_wrap_array = np.where(np.abs(np.diff(y)) > high / 2.0)[0]
    id_min = 0
    for id_wrap in id_wrap_array:
        # Step size in x direction
        dx = x[id_wrap+1] - x[id_wrap]

        # Wrap around 0 deg
        if np.diff(y)[id_wrap] > high / 2.0:  
            dy = y[id_wrap] - y[id_wrap + 1] + high
            xtp = x[id_wrap] + dx * (y[id_wrap]) / dy  # transition point
            xw = np.hstack([xw, x[id_min:id_wrap + 1], xtp - 0.001 * dx, xtp, xtp + 0.001 * dx])
            yw = np.hstack([yw, y[id_min:id_wrap + 1], 0.0, np.nan, high])

        # Wrap around 360 deg
        elif np.diff(y)[id_wrap] < - high / 2.0:
            dy = y[id_wrap+1] - y[id_wrap] + high
            xtp = x[id_wrap] + dx * (high - y[id_wrap]) / dy  # transition point
            xw = np.hstack([xw, x[id_min:id_wrap + 1], xtp - 0.001, xtp, xtp + 0.001])
            yw = np.hstack([yw, y[id_min:id_wrap + 1], high, np.nan, 0.0])

        id_min = id_wrap + 1

    # Append remaining data
    xw = np.hstack([xw, x[id_min::]])
    yw = np.hstack([yw, y[id_min::]])

    # Now plot lines, without markers
    if (marker is None):
        ax.plot(xw, yw, linestyle=linestyle, color=color, label=label)  # Plot without marker, but with label
    else:
        ax.plot(xw, yw, linestyle=linestyle, color=color)  # Plot lines, without markers
        ax.scatter(x, y, marker=marker, color=color)  # Now plot markers, only at non-transition points

        # Now add a placeholder (empty) line with right marker for the legend
        if label is not None:
            ax.plot(
                xw[0:0],
                yw[0:0],
                linestyle=linestyle,
                marker=marker,
                label=label,
                color=color
            )

    return ax



if __name__ == "__main__":
    # Demonstrate use case
    y = 180.0 + 200.0 * np.sin(np.linspace(0.0, 10.0, 50))
    y = np.remainder(y, 360.0)
    t = np.arange(len(y))

    # Create figure and produce plots using pyplot and using flasc
    fig, ax = plt.subplots()
    ax.plot(t, y, color="gray", label="Raw")
    plot_with_wrapping(t, y, ax=ax, high=360.0, color="orange", marker="o", label="Wrapped")

    # Format plot and show
    ax.grid(True)
    ax.legend()
    ax.set_ylim([0, 360.0])
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind direction (deg)")

    plt.show()
