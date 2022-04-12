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
    color=None,
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

    if color is None:
        # Use matplotlib's internal color cycler
        color = ax._get_lines.prop_cycler.__next__()['color']

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
            xw = np.hstack([xw, x[id_min:id_wrap + 1], xtp - 0.001 * dx, xtp, xtp + 0.001 * dx])
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


def plot_floris_layout(fi, turbine_names=None, plot_terrain=True):
    """Plot the wind farm layout and turbine performance curves for the
    floris object of interest. This visualization function includes some
    useful checks such as checking which turbine curves are identical,
    and then plot those accordingly. It also includes the plotting of
    different hub heights through a background colormap.

    Args:
        fi (FlorisInterface): The FLORIS object
        turbine_names (iteratible, optional): List of turbine names, with
        each entry being a string. It is recommended that this is something
        like one or two letters, and then a number to indicate the turbine.
        For example, A01, A02, A03, ... If None is specified, will assume
        turbine names T01, T02, T03, .... Defaults to None.
        plot_terrain (bool, optional): Plot the terrain as a colormap.
        Defaults to True.

    Returns:
        _type_: _description_
    """
    # Plot turbine configurations
    fig = plt.figure(figsize=(16, 8))

    if turbine_names is None:
        nturbs = len(fi.layout_x)
        turbine_names = ["T{:02d}".format(ti) for ti in range(nturbs)]

    plt.subplot(1, 2, 1)
    ax = [None, None, None]
    ax[0] = plt.gca()

    if plot_terrain:
        hub_heights = fi.floris.farm.hub_heights.flatten()
        cntr = ax[0].tricontourf(
            fi.layout_x,
            fi.layout_y,
            hub_heights,
            levels=14,
            cmap="RdBu_r"
        )
        fig.colorbar(
            cntr,
            ax=ax[0],
            label='Terrain-corrected hub height (m)',
            ticks=np.linspace(
                np.min(hub_heights) - 10.0,
                np.max(hub_heights) + 10.0,
                15,
            )
        )

    turbine_types = (
        [t["turbine_type"] for t in fi.floris.farm.turbine_definitions]
    )
    turbine_types = np.array(turbine_types, dtype="str")
    for tt in np.unique(turbine_types):
        ids = (turbine_types == tt)
        ax[0].plot(fi.layout_x[ids], fi.layout_y[ids], "o", label=tt)

    # Plot turbine names and hub heights
    for ti in range(len(fi.layout_x)):
        ax[0].text(
            fi.layout_x[ti],
            fi.layout_y[ti],
            turbine_names[ti] + " ({:.1f} m)".format(hub_heights[ti])
        )

    ax[0].axis("equal")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_xlabel("x coordinate (m)")
    ax[0].set_ylabel("y coordinate (m)")
    ax[0].set_title("Farm layout")

    # Plot turbine power and thrust curves
    plt.subplot(2, 2, 2)
    ax[1] = plt.gca()
    plt.subplot(2, 2, 4)
    ax[2] = plt.gca()
    # Identify unique power-thrust curves and group turbines accordingly
    for ti in range(len(fi.layout_x)):
        pt = fi.floris.farm.turbine_definitions[ti]["power_thrust_table"]
        if ti == 0:
            unique_pt = [pt]
            unique_turbines = [[ti]]
            continue

        # Check if power-thrust curve already exists somewhere
        is_unique = True
        for tii in range(len(unique_pt)):
            if (unique_pt[tii] == pt):
                unique_turbines[tii].append(ti)
                is_unique = False
                continue

        # If not, append as new entry
        if is_unique:
            unique_pt.append(pt)
            unique_turbines.append([ti])

    for tii, pt in enumerate(unique_pt):
        # Convert a very long string of turbine identifiers to ranges,
        # e.g., from "A01, A02, A03, A04" to "A01-A04"
        labels = [turbine_names[i] for i in unique_turbines[tii]]
        prev_turb_in_list = np.zeros(len(labels), dtype=bool)
        next_turb_in_list = np.zeros(len(labels), dtype=bool)
        for ii, lb in enumerate(labels):
            # Split initial string from sequence of texts
            idx = 0
            while lb[0:idx+1].isalpha():
                idx += 1
            
            # Now check various choices of numbers, i.e., A001, A01, A1
            turb_prev_if_range = [
                lb[0:idx] + "{:01d}".format(int(lb[idx::]) - 1),
                lb[0:idx] + "{:02d}".format(int(lb[idx::]) - 1),
                lb[0:idx] + "{:03d}".format(int(lb[idx::]) - 1)
            ]
            turb_next_if_range = [
                lb[0:idx] + "{:01d}".format(int(lb[idx::]) + 1),
                lb[0:idx] + "{:02d}".format(int(lb[idx::]) + 1),
                lb[0:idx] + "{:03d}".format(int(lb[idx::]) + 1)
            ]

            prev_turb_in_list[ii] = np.any([t in labels for t in turb_prev_if_range])
            next_turb_in_list[ii] = np.any([t in labels for t in turb_next_if_range])

        # Remove label for turbines in the middle of ranges
        for id in np.where(prev_turb_in_list & next_turb_in_list)[0]:
            labels[id] = ""

        # Append a dash to labels for turbines at the start of a range
        for id in np.where(~prev_turb_in_list & next_turb_in_list)[0]:
            labels[id] += "-"

        # Append a comma to turbines at the end of a range
        for id in np.where(~next_turb_in_list)[0]:
            labels[id] += ","

        # Now join all strings to a single label and remove last comma
        label = "".join(labels)[0:-1]

        # Plot power and thrust curves for groups of turbines
        tn = fi.floris.farm.turbine_definitions[unique_turbines[tii][0]]["turbine_type"]
        ax[1].plot(pt["wind_speed"], pt["power"], label=label + " ({:s})".format(tn))
        ax[2].plot(pt["wind_speed"], pt["thrust"], label=label + " ({:s})".format(tn))

        ax[1].set_xlabel("Wind speed (m/s)")
        ax[2].set_xlabel("Wind speed (m/s)")
        ax[1].set_ylabel("Power coefficient (-)")
        ax[2].set_ylabel("Thrust coefficient (-)")
        ax[1].grid(True)
        ax[2].grid(True)
        ax[1].legend()
        ax[2].legend()

    return fig, ax
