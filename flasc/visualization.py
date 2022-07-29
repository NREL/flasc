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
    label=None,
    rasterized=False,
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
        rasterize (bool, optional): Rasterize the plot/line and thereby remove
        its vectorized properties. This can help reduce the size of a .pdf or
        .eps file when this figure is saved, at the cost of a potential
        reduction in image quality.

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
        ax.plot(xw, yw, linestyle=linestyle, color=color, label=label, rasterized=rasterized)
    else:
        # Plot lines, without markers
        ax.plot(xw, yw, linestyle=linestyle, color=color, rasterized=rasterized)
        # Now plot markers, only at non-transition points
        ax.scatter(x, y, marker=marker, color=color, rasterized=rasterized)

        # Now add a placeholder (empty) line with right marker for the legend
        if label is not None:
            ax.plot(
                xw[0:0],
                yw[0:0],
                linestyle=linestyle,
                marker=marker,
                label=label,
                color=color,
                rasterized=rasterized
            )

    return ax


def plot_floris_layout(fi, plot_terrain=True):
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
    
    ax = [None, None, None]
    ax[0] = fig.add_subplot(121)

    if plot_terrain:
        plot_farm_terrain(fi, fig, ax[0])

    # Generate plotting dictionary based on turbine; plot locations
    turbine_types = (
        [t["turbine_type"] for t in fi.floris.farm.turbine_definitions]
    )
    turbine_types = np.array(turbine_types, dtype="str")
    for ti, tt in enumerate(np.unique(turbine_types)):
        plotting_dict = {
            "turbine_indices" : np.array(range(len(fi.layout_x)))\
                [turbine_types == tt],
            "turbine_names" : generate_labels_with_hub_heights(fi),
            "color" : "C%s" % ti,
            "label" : tt
        }
        plot_layout_only(fi, plotting_dict, ax=ax[0])
    ax[0].legend()

    # Power and thrust curve plots
    ax[1] = fig.add_subplot(222)
    ax[2] = fig.add_subplot(224)

    # Identify unique power-thrust curves and group turbines accordingly
    unique_turbine_types, utt_ids = np.unique(turbine_types, return_index=True)
    for ti, (tt, tti) in enumerate(zip(unique_turbine_types, utt_ids)):
        pt = fi.floris.farm.turbine_definitions[tti]["power_thrust_table"]

        plotting_dict = {
            "color" : "C%s" % ti,
            "label" : tt
        }
        plot_power_curve_only(pt, plotting_dict, ax=ax[1])
        plot_thrust_curve_only(pt, plotting_dict, ax=ax[2])

    return fig, ax

def generate_default_labels(fi):
    labels = ["T{0:02d}".format(ti) for ti in range(len(fi.layout_x))]
    return labels

def generate_labels_with_hub_heights(fi):
    labels = ["T{0:02d} ({1:.1f} m)".format(ti, h) for ti, h in 
        enumerate(fi.floris.farm.hub_heights.flatten())]
    return labels

def plot_layout_only(fi, plotting_dict={}, ax=None):
    """
    Inputs:
    - plotting_dict: dictionary of plotting parameters, with the 
        following (optional) fields and their (default) values:
            "turbine_indices" : (range(len(fi.layout_x))) (turbines to 
                                plot, default to all turbines)
            "turbine_names" : (["TX" for X in range(len(fi.layout_x)])
            "color" : ("black")
            "marker" : (".")
            "markersize" : (10)
            "label" : (None) (for legend, if desired)
    - ax: axes to plot on (if None, creates figure and axes)
    - NOTE: turbine_names should be a complete list of all turbine names; only
            those in turbine_indeces will be plotted though.
    """

    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    # Generate plotting dictionary
    default_plotting_dict = {
        "turbine_indices" : range(len(fi.layout_x)),
        "turbine_names" : generate_default_labels(fi),
        "color" : "black", 
        "marker" : ".",
        "markersize" : 10,
        "label" : None
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}
    if len(plotting_dict["turbine_names"]) == 0: # empty list provided
        plotting_dict["turbine_names"] = [""]*len(fi.layout_x)

    # Plot
    ax.plot(
        fi.layout_x[plotting_dict["turbine_indices"]], 
        fi.layout_y[plotting_dict["turbine_indices"]],
        marker=plotting_dict["marker"],
        markersize=plotting_dict["markersize"],
        linestyle="None",
        color=plotting_dict["color"],
        label=plotting_dict["label"]
    )

    # Add labels to plot, if desired
    for ti in plotting_dict["turbine_indices"]:
       ax.text(fi.layout_x[ti], fi.layout_y[ti], 
           plotting_dict["turbine_names"][ti])

    # Plot labels and aesthetics
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    ax.set_title("Farm layout")

    return ax

def plot_power_curve_only(pt, plotting_dict={}, ax=None):
    """
    pt expected to have keys "wind_speed" and "power"
    """
    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    default_plotting_dict = {
        "color" : "black", 
        "marker" : ".",
        "markersize" : 10,
        "label" : None
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # Plot power and thrust curves for groups of turbines
    ax.plot(pt["wind_speed"], pt["power"], color=plotting_dict["color"], 
        label=plotting_dict["label"])
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power coefficient (-)")
    ax.set_xlim([pt["wind_speed"][0], pt["wind_speed"][-1]])
    ax.grid(True)

    return ax

def plot_thrust_curve_only(pt, plotting_dict, ax=None):
    """
    pt expected to have keys "wind_speed" and "thrust"
    """
    
    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    default_plotting_dict = {
        "color" : "black", 
        "marker" : ".",
        "markersize" : 10,
        "label" : None
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # Plot power and thrust curves for groups of turbines
    ax.plot(pt["wind_speed"], pt["thrust"], color=plotting_dict["color"], 
        label=plotting_dict["label"])
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Thrust coefficient (-)")
    ax.set_xlim([pt["wind_speed"][0], pt["wind_speed"][-1]])
    ax.grid(True)

    return ax

def plot_farm_terrain(fi, fig, ax):
    hub_heights = fi.floris.farm.hub_heights.flatten()
    cntr = ax.tricontourf(
        fi.layout_x,
        fi.layout_y,
        hub_heights,
        levels=14,
        cmap="RdBu_r"
    )
    
    fig.colorbar(
        cntr,
        ax=ax,
        label='Terrain-corrected hub height (m)',
        ticks=np.linspace(
            np.min(hub_heights) - 10.0,
            np.max(hub_heights) + 10.0,
            15,
        )
    )