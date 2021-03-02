import pandas as pd
from floris.utilities import wrap_180, wrap_360
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import ephem

# Simple function to display the size of a data frame
def print_size(df):
    print('Num rows in df: %d' % df.shape[0])

# Define an angular mean function, which computes the angular mean
# of all the columns of a dataframe, given in degrees
def angular_mean(df):
    cos_mean = np.cos(np.deg2rad(df)).mean(axis=1,skipna=True)
    sin_mean = np.sin(np.deg2rad(df)).mean(axis=1,skipna=True)
    df['wd'] = wrap_360(np.rad2deg(np.arctan2(sin_mean,cos_mean)))
    return df['wd']

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_daytime(df, lat, long, elevation,threshold = -8):
    
    # Set up daylight stuff
    sun = ephem.Sun()
    observer = ephem.Observer()
    observer.lat, observer.lon, observer.elevation = str(lat), str(long), int(elevation)
    # observer.lat, observer.lon, observer.elevation = '40.995', '-103.524', 932

    # Determine sun_alt and daytime
    df['sun_alt'] = np.nan
    df['is_daytime'] = 1
    for idx in df.index:
        observer.date = df.loc[idx,'time']
        sun.compute(observer)
        df.loc[idx,'sun_alt'] = sun.alt * 180 / np.pi
    df.loc[df.sun_alt < -8,'is_daytime']=0 
    return df

def data_plot(
    x,
    y,
    color="b",
    label="_nolegend_",
    x_bins=None,
    x_radius=None,
    ax=None,
    show_scatter=True,
    show_bin_points=True,
    show_confidence=True,
    ls = '-',
    marker = 'None'
):
    """
    To do...
    """
    if not ax:
        fig, ax = plt.subplots()

    df = pd.DataFrame({"x": x, "y": y})

    if df.shape[0] > 0:

        # If bins not provided, just use ints
        if x_bins is None:
            x_bins = np.arange(df["x"].astype(int).min(), df["x"].astype(int).max(), 1)

        # if no radius provided, use bins to determine
        if x_radius is None:
            x_radius = (x_bins[1] - x_bins[0]) / 2.0

        # now loop over bins and determine stats
        mean_vals = np.zeros_like(x_bins) * np.nan
        count_vals = np.zeros_like(x_bins) * np.nan
        lower = np.zeros_like(x_bins) * np.nan
        upper = np.zeros_like(x_bins) * np.nan

        for x_idx, x_cent in enumerate(x_bins):

            df_sub = df[(df.x >= x_cent - x_radius) & (df.x <= x_cent + x_radius)]


            count_vals[x_idx] = df_sub.shape[0]

            confidence = 0.95
            mean_vals[x_idx],lower[x_idx],upper[x_idx]  =  mean_confidence_interval(df_sub.y.values, confidence=confidence)
 

        # # Plot the underlying points
        if show_scatter:
            ax.scatter(
                df["x"],
                df["y"],
                color=color,
                label="_nolegend_",
                alpha=1.0,
                s=35,
                marker=".",
            )
        if show_bin_points:
            ax.scatter(
                x_bins,
                mean_vals,
                color=color,
                s=count_vals,
                label="_nolegend_",
                alpha=0.6,
                marker="s",
            )

        ax.plot(x_bins, mean_vals, label=label, color=color, ls=ls, marker=marker)


        if show_confidence:
            ax.fill_between(
                x_bins, lower, upper, alpha=0.2, color=color, label="_nolegend_"
            )


        return x_bins, mean_vals, lower, upper

    else:
        ax.plot(0, 0, label=label, color=color)

        return np.nan, np.nan, np.nan, np.nan
