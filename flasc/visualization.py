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


def plot_with_wrapping(
    x,
    y,
    ax=None,
    low=0.0,
    high=360.0,
    linestyle="-",
    marker=None,
    color="black",
    label=None
):
    """Plot a line on an axis that deals with angle wrapping. Normally, using
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

    Returns:
        ax: Axis object of the matplotlib.pyplot class on which the line (and
        optionally markers) are plotted.
    """

    # Create figure, if not provided
    if ax is None:
        fig, ax = plt.subplots()

    if (low >= high):
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
        dx = x[id_wrap+1] - x[id_wrap]

        # Wrap around 0 deg
        if np.diff(y)[id_wrap] > high_norm / 2.0:  
            dy = y[id_wrap] - y[id_wrap + 1] + high_norm
            xtp = x[id_wrap] + dx * (y[id_wrap]) / dy  # transition point
            xw = np.hstack([xw, x[id_min:id_wrap + 1], xtp - 0.001 * dx, xtp, xtp + 0.001 * dx])
            yw = np.hstack([yw, y[id_min:id_wrap + 1], 0.0, np.nan, high_norm])

        # Wrap around 360 deg
        elif np.diff(y)[id_wrap] < - high_norm / 2.0:
            dy = y[id_wrap+1] - y[id_wrap] + high_norm
            xtp = x[id_wrap] + dx * (high_norm - y[id_wrap]) / dy  # transition point
            xw = np.hstack([xw, x[id_min:id_wrap + 1], xtp - 0.001, xtp, xtp + 0.001])
            yw = np.hstack([yw, y[id_min:id_wrap + 1], high_norm, np.nan, 0.0])

        id_min = id_wrap + 1

    # Append remaining data
    xw = np.hstack([xw, x[id_min::]])
    yw = np.hstack([yw, y[id_min::]])

    # Reintroduce offset from 'low'
    yw = yw + low
    y = y + low

    # Now plot lines, without markers
    if (marker is None):
        # Plot without marker, but with label
        ax.plot(xw, yw, linestyle=linestyle, color=color, label=label)
    else:
        # Plot lines, without markers
        ax.plot(xw, yw, linestyle=linestyle, color=color)
        # Now plot markers, only at non-transition points
        ax.scatter(x, y, marker=marker, color=color)

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
    for ii in range(2):
        if ii == 0:
            # Do case with compass directions (from 0 to +360 deg)
            y = 180 + 200.0 * np.sin(np.linspace(0.0, 10.0, 100))
            y = np.remainder(y, 360.0)
            low = 0.0
            high = 360.0
        if ii == 1:
            # Do case with vane directions (from -180 to +180 deg)
            y = 200.0 * np.sin(np.linspace(0.0, 10.0, 100))
            y[y < -180.0] += 360.0
            y[y >  180.0] -= 360.0
            low = -180.0
            high = 180.0


        t = np.arange(len(y))

        # Create figure and produce plots using pyplot and using flasc
        fig, ax = plt.subplots()
        ax.plot(t, y, color="gray", label="Raw")
        plot_with_wrapping(t, y, ax=ax, low=low, high=high, color="orange", marker="o", label="Wrapped")

        # Format plot and show
        ax.grid(True)
        ax.legend()
        ax.set_ylim([low - 10.0, high + 10.0])
        ax.set_xlabel("Time")
        ax.set_ylabel("Direction wrapped around \n low={:.1f}, high={:.1f} (deg)".format(low, high))
        plt.tight_layout()

    plt.show()
