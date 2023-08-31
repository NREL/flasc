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
import pandas as pd
import scipy.stats as st


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
        turbine names T01, T02, T03, .... Defaults to None. To avoid printing
        names, specify turbine_names=[].
        plot_terrain (bool, optional): Plot the terrain as a colormap.
        Defaults to True.

    Returns:
        _type_: _description_
    """
    # Plot turbine configurations
    fig = plt.figure(figsize=(16, 8))

    # Get names if not provided
    if turbine_names is None:
        turbine_names = generate_labels_with_hub_heights(fi)
    
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
            "turbine_names" : turbine_names,
            "color" : "C%s" % ti,
            "label" : tt
        }
        plot_layout_only(fi, plotting_dict, ax=ax[0])
    ax[0].legend()
    ax[0].set_title("Farm layout")

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
    Plot the farm layout.

    Args:
        plotting_dict: dictionary of plotting parameters, with the 
            following (optional) fields and their (default) values:
            "turbine_indices" : (range(len(fi.layout_x))) (turbines to 
                                plot, default to all turbines)
            "turbine_names" : (["TX" for X in range(len(fi.layout_x)])
            "color" : ("black")
            "marker" : (".")
            "markersize" : (10)
            "label" : (None) (for legend, if desired)
        ax: axes to plot on (if None, creates figure and axes)
    
    Returns:
        ax: the current axes for the layout plot

    turbine_names should be a complete list of all turbine names; only
    those in turbine_indices will be plotted though.
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

    return ax

def plot_power_curve_only(pt, plotting_dict={}, ax=None):
    """
    Generate plot of turbine power curve. 

    Args:
        pt: power-thrust table as a dictionary. Expected to contain 
            keys "wind_speed" and "power"
        plotting_dict: dictionary of plotting parameters, with the 
            following (optional) fields and their (default) values:
            "color" : ("black"), 
            "linestyle" : ("solid"),
            "linewidth" : (2),
            "label" : (None)
        ax: axes to plot on (if None, creates figure and axes)
    
    Returns:
        ax: the current axes for the power curve plot
    """
    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    default_plotting_dict = {
        "color" : "black", 
        "linestyle" : "solid",
        "linewidth" : 2,
        "label" : None
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # Plot power and thrust curves for groups of turbines
    ax.plot(pt["wind_speed"], pt["power"], **plotting_dict)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power coefficient (-)")
    ax.set_xlim([pt["wind_speed"][0], pt["wind_speed"][-1]])
    ax.grid(True)

    return ax

def plot_thrust_curve_only(pt, plotting_dict, ax=None):
    """
    Generate plot of turbine thrust curve. 

    Args:
        pt: power-thrust table as a dictionary. Expected to contain 
            keys "wind_speed" and "thrust"
        plotting_dict: dictionary of plotting parameters, with the 
            following (optional) fields and their (default) values:
            "color" : ("black"), 
            "linestyle" : ("solid"),
            "linewidth" : (2),
            "label" : (None)
        ax: axes to plot on (if None, creates figure and axes)
    
    Returns:
        ax: the current axes for the thrust curve plot
    """
    
    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    default_plotting_dict = {
        "color" : "black", 
        "linestyle" : "solid",
        "linewidth" : 2,
        "label" : None
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # Plot power and thrust curves for groups of turbines
    ax.plot(pt["wind_speed"], pt["thrust"], **plotting_dict)
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

def plot_layout_with_waking_directions(
    fi, 
    layout_plotting_dict={}, 
    wake_plotting_dict={},
    D=None,
    limit_dist_D=None,
    limit_dist_m=None,
    limit_num=None,
    wake_label_size=7,
    ax=None
    ):
    """
    Plot waking directions and distances between turbines.

    Args:
        fi: Instantiated FlorisInterface object
        layout_plotting_dict: dictionary of plotting parameters for 
            turbine locations. Defaults to the defaults of 
            plot_layout_only.
        wake_plotting_dict: dictionary of plotting parameters for the 
            waking directions, with the following (optional) fields and 
            their (default) values:
            "color" : ("black"), 
            "linestyle" : ("solid"),
            "linewidth" : (0.5)
        D: rotor diameter. Defaults to the rotor diamter of the first 
            turbine in the Floris object.
        limit_dist_D: limit on the distance between turbines to plot, 
            specified in rotor diamters.
        limit_dist_m: limit on the distance between turbines to plot, 
            specified in meters. If specified, overrides limit_dist_D.
        limit_num: limit on number of outgoing neighbors to include. 
            If specified, only the limit_num closest turbines are 
            plotted. However, directions already plotted from other 
            turbines are not considered in the count.
        wake_label_size: font size for labels of direction/distance.
        ax: axes to plot on (if None, creates figure and axes)
    
    Returns:
        ax: the current axes for the thrust curve plot
    """
    
    # Combine default plotting options
    def_wake_plotting_dict = {
        "color" : "black", 
        "linestyle" : "solid",
        "linewidth" : 0.5,
    }
    wake_plotting_dict = {**def_wake_plotting_dict, **wake_plotting_dict}
    
    def_layout_plotting_dict = {"turbine_indices" : range(len(fi.layout_x))}
    layout_plotting_dict = {**def_layout_plotting_dict, **layout_plotting_dict}

    ax = plot_layout_only(fi, plotting_dict=layout_plotting_dict, ax=ax)

    N_turbs = len(fi.floris.farm.turbine_definitions)
    
    if D is None:
        D = fi.floris.farm.turbine_definitions[0]['rotor_diameter']
        # TODO: build out capability to use multiple diameters, if of interest.
        # D = np.array([turb['rotor_diameter'] for turb in 
        #      fi.floris.farm.turbine_definitions])
    #else:
        #D = D*np.ones(N_turbs)

    dists_m = np.zeros((N_turbs, N_turbs))
    angles_d = np.zeros((N_turbs, N_turbs))

    for i in range(N_turbs):
        for j in range(N_turbs):
            dists_m[i,j] = np.linalg.norm(
                [fi.layout_x[i]-fi.layout_x[j], fi.layout_y[i]-fi.layout_y[j]]
            )
            angles_d[i,j] = wake_angle(
                fi.layout_x[i], fi.layout_y[i], fi.layout_x[j], fi.layout_y[j]
            )

    # Mask based on the limit distance (assumed to be in measurement D)
    if limit_dist_D is not None and limit_dist_m is None:
        limit_dist_m = limit_dist_D * D
    if limit_dist_m is not None:
        mask = dists_m > limit_dist_m
        dists_m[mask] = np.nan
        angles_d[mask] = np.nan

    # Handle default limit number case
    if limit_num is None:
        limit_num = -1

    # Loop over pairs, plot
    label_exists = np.full((N_turbs, N_turbs), False)
    for i in range(N_turbs):
        for j in range(N_turbs):
            #import ipdb; ipdb.set_trace()
            if ~np.isnan(dists_m[i, j]) and \
                dists_m[i, j] != 0.0 and \
                ~(dists_m[i, j] > np.sort(dists_m[i,:])[limit_num]) and \
                i in layout_plotting_dict["turbine_indices"] and \
                j in layout_plotting_dict["turbine_indices"]:

                (l,) = ax.plot(fi.layout_x[[i,j]], fi.layout_y[[i,j]],
                               **wake_plotting_dict)

                # Only label in one direction
                if ~label_exists[i,j]:
               
                    linetext = "{0:.1f} D --- {1:.0f}/{2:.0f}".format(
                        dists_m[i,j] / D,
                        angles_d[i,j], 
                        angles_d[j,i],
                    )

                    label_line(
                        l, linetext, ax, near_i=1, near_x=None, near_y=None, 
                        rotation_offset=0, size=wake_label_size
                    )

                    label_exists[i,j] = True
                    label_exists[j,i] = True

    return ax
    
def wake_angle(x_i, y_i, x_j, y_j):
    """
    Get angles between turbines in wake direction

    Args:
        x_i: x location of turbine i
        y_i: y location of turbine i
        x_j: x location of turbine j
        y_j: y location of turbine j
        
    Returns:
        wakeAngle (float): angle between turbines relative to compass
    """
    wakeAngle = (
        np.arctan2(y_i - y_j, x_i - x_j) * 180.0 / np.pi
    )  # Angle in normal cartesian coordinates

    # Convert angle to compass angle
    wakeAngle = 270.0 - wakeAngle
    if wakeAngle < 0:
        wakeAngle = wakeAngle + 360.0
    if wakeAngle > 360:
        wakeAngle = wakeAngle - 360.0

    return wakeAngle

def label_line(
    line,
    label_text,
    ax,
    near_i=None,
    near_x=None,
    near_y=None,
    rotation_offset=0.0,
    offset=(0, 0),
    size=7,
):
    """
    [summary]

    Args:
        line (matplotlib.lines.Line2D): line to label.
        label_text (str): label to add to line.
        ax (:py:class:`matplotlib.pyplot.axes` optional): figure axes.
        near_i (int, optional): Catch line near index i.
            Defaults to None.
        near_x (float, optional): Catch line near coordinate x.
            Defaults to None.
        near_y (float, optional): Catch line near coordinate y.
            Defaults to None.
        rotation_offset (float, optional): label rotation in degrees.
            Defaults to 0.
        offset (tuple, optional): label offset from turbine location.
            Defaults to (0, 0).
        size (float): font size. Defaults to 7.

    Raises:
        ValueError: ("Need one of near_i, near_x, near_y") raised if
            insufficient information is passed in.
    """

    def put_label(i):
        """
        Add a label to index.

        Args:
            i (int): index to label.
        """
        i = min(i, len(x) - 2)
        dx = sx[i + 1] - sx[i]
        dy = sy[i + 1] - sy[i]
        rotation = np.rad2deg(np.arctan2(dy, dx)) + rotation_offset
        pos = [(x[i] + x[i + 1]) / 2.0 + offset[0], (y[i] + y[i + 1]) / 2 + offset[1]]
        plt.text(
            pos[0],
            pos[1],
            label_text,
            size=size,
            rotation=rotation,
            color=line.get_color(),
            ha="center",
            va="center",
            bbox=dict(ec="1", fc="1", alpha=0.8),
        )

    # extract line data
    x = line.get_xdata()
    y = line.get_ydata()

    # define screen spacing
    if ax.get_xscale() == "log":
        sx = np.log10(x)
    else:
        sx = x
    if ax.get_yscale() == "log":
        sy = np.log10(y)
    else:
        sy = y

    # find index
    if near_i is not None:
        i = near_i
        if i < 0:  # sanitize negative i
            i = len(x) + i
        put_label(i)
    elif near_x is not None:
        for i in range(len(x) - 2):
            if (x[i] < near_x and x[i + 1] >= near_x) or (
                x[i + 1] < near_x and x[i] >= near_x
            ):
                put_label(i)
    elif near_y is not None:
        for i in range(len(y) - 2):
            if (y[i] < near_y and y[i + 1] >= near_y) or (
                y[i + 1] < near_y and y[i] >= near_y
            ):
                put_label(i)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")

def shade_region(points, show_points=False, plotting_dict_region={}, 
    plotting_dict_points={}, ax=None):
    """
    Shade a region defined by a series of vertices (points).

    Args:
        points: 2D array of vertices for the shaded region, shape N x 2, 
            where each row contains a coordinate (x, y)
        show_points: Boolean to dictate whether to plot the points as well 
            as the shaded region
        plotting_dict_region: dictionary of plotting parameters for the shaded
            region, with the following (optional) fields and their (default) 
            values:
            "color" : ("black")
            "edgecolor": (None)
            "alpha" : (0.3)
            "label" : (None) (for legend, if desired)
        plotting_dict_region: dictionary of plotting parameters for the 
            vertices (points), with the following (optional) fields and their 
            (default) values:
            "color" : "black", 
            "marker" : (None)
            "s" : (10)
            "label" : (None) (for legend, if desired)
        ax: axes to plot on (if None, creates figure and axes)
    
    Returns:
        ax: the current axes for the layout plot
    """

    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    # Generate plotting dictionary
    default_plotting_dict_region = {
        "color" : "black", 
        "edgecolor" : None,
        "alpha" : 0.3,
        "label" : None
    }
    plotting_dict_region = {**default_plotting_dict_region,
                            **plotting_dict_region}

    ax.fill(points[:,0], points[:,1], **plotting_dict_region)

    if show_points:
        default_plotting_dict_points = {
            "color" : "black", 
            "marker" : ".",
            "s" : 10,
            "label" : None
        }
        plotting_dict_points = {**default_plotting_dict_points, 
                                **plotting_dict_points}

        ax.scatter(points[:,0], points[:,1], **plotting_dict_points)

    # Plot labels and aesthetics
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    if plotting_dict_region["label"] is not None or \
        plotting_dict_points["label"] is not None:
        ax.legend()

    return ax


def plot_binned_mean_and_ci(
    x,
    y,
    color="b",
    label="_nolegend_",
    x_edges=None,
    ax=None,
    show_scatter=True,
    show_bin_points=True,
    show_confidence=True,
    alpha_scatter=0.1,
    confidence_level = 0.95,
):
    """
    Plot data to a single axis.  Method
    has options to include scatter of underlying data, specifiying
    bin edges, and plotting confidence interval.

    Args:
        x (np.array): abscissa data.
        y (np.array): ordinate data.
        color (str, optional): line color.
            Defaults to 'b'.
        label (str, optional): line label used in legend.
            Defaults to '_nolegend_'.
        x_edges (np.array, optional): bin edges in x data
            Defaults to None.
        ax (:py:class:`matplotlib.pyplot.axes`, optional):
            axes handle for plotting. Defaults to None.
        show_scatter (bool, optional): flag to control scatter plot.
            Defaults to True.
        show_bin_points (bool, optional): flag to control plot of bins.
            Defaults to True.
        show_confidence (bool, optional): flag to control plot of
            confidence interval. Defaults to True.
        alpha_scatter (float, optional): Alpha for scatter
            plot. Defaults to 0.5.
        confidenceLevel (float, optional): Confidence level for
            confidence interval. Defaults to 0.95.

    """

    # Check the length of x equals length of y
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    
    # Check that x is not empty
    if len(x) == 0:
        raise ValueError("x is empty")
    

    # Declare ax if not provided
    if ax is None:
        _, ax = plt.subplots()

    # Put points ino dataframe
    df = pd.DataFrame({"x": x, "y": y})

    # If x_edges not provided, use 50 bins over range of x
    if x_edges is None:
        x_edges = np.linspace(df["x"].min()*.98, df["x"].max()*1.02, 50)

    # Define x_labels as bin centers
    x_labels = (x_edges[1:] + x_edges[:-1]) / 2.0

    # Bin data
    df["x_bin"] = pd.cut(df["x"], x_edges, labels=x_labels)

    # Get aggregate statistics
    df_agg = df.groupby("x_bin").agg(
        {"y": ["count", "std", "min", "max", "mean", st.sem]}
    )
    # Flatten column names
    df_agg.columns = ["_".join(c) for c in df_agg.columns]

    # Reset the index
    df_agg = df_agg.reset_index()

    # Delete rows with no data
    df_agg = df_agg[df_agg["y_count"] > 0]

    # Add the confidence interval of the mean to df_agg
    df_agg["y_ci_lower"], df_agg["y_ci_upper"] = st.t.interval(
        confidence_level, 
        df_agg["y_count"]-1,
        loc=df_agg["y_mean"],
        scale=df_agg["y_sem"]
    )

    # Plot the mean values
    ax.plot(df_agg.x_bin, df_agg.y_mean, color=color, label=label)

    # Plot the confidence interval
    if show_confidence:
        ax.fill_between(
            df_agg.x_bin,
            df_agg.y_ci_lower,
            df_agg.y_ci_upper,
            color=color,
            alpha=0.2,
        )

        # Plot a dasshed line at confidence interval
        ax.plot(
            df_agg.x_bin,
            df_agg.y_ci_lower,
            color=color,
            alpha=0.2,
            ls="--",
        )
        ax.plot(
            df_agg.x_bin,
            df_agg.y_ci_upper,
            color=color,
            alpha=0.2,
            ls="--",
        )

    # Plot the scatter points
    if show_scatter:
        ax.scatter(df.x, df.y, color=color, s=10, alpha=alpha_scatter)

    # Plot the bin points, scaled by the counts
    if show_bin_points:
        ax.scatter(
            df_agg.x_bin,
            df_agg.y_mean,
            color=color,
            s=df_agg.y_count / df_agg.y_count.max() * 20,
            alpha=0.5,
            marker='s'
        )

    return ax