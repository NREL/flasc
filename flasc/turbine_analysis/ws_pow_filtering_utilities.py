# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import scipy.stats as scst

from ..dataframe_operations import dataframe_manipulations as dfm
from .. import utilities as fsut


def estimate_rated_powers_from_data(df):
    nturbs = dfm.get_num_turbines(df)
    est_ratedpw_list = np.zeros(nturbs)
    for ti in range(nturbs):
        rated_ids = (df['ws_%03d' % ti] > 15.)
        df_subset = df.loc[rated_ids, 'pow_%03d' % (ti)]
        if df_subset.shape[0] > 0:
            est_ratedpw = np.nanmedian(
                df.loc[rated_ids, 'pow_%03d' % (ti)])
        else:
            est_ratedpw = np.nan
        if np.isnan(est_ratedpw):
            est_ratedpw = 1.  # Placeholder
        elif est_ratedpw < 20.0:
            est_ratedpw = np.round(est_ratedpw, 1)  # MW
        elif est_ratedpw < 20.0e3:
            est_ratedpw = np.round(est_ratedpw/1e3, 1)*1e3  # kW
        else:
            est_ratedpw = np.round(est_ratedpw/1e6, 1)*1e6  # W
        est_ratedpw_list[ti] = float(est_ratedpw)

    return est_ratedpw_list


def _approximate_large_scatter_plot(x, y, N=100, bounds=None):
    # This function bins all the data is x- and y-direction and then
    # returns and (x, y) output that lists the centers of the non-empty
    # bins. This particularly useful when trying to plot millions of
    # data points on top of each other -- it saves orders of magnitude
    # on plotting and saving figures. Also, pdfs produced in this manner
    # Are much lighter to handle.

    x = np.asarray(x, dtype='float')
    y = np.asarray(y, dtype='float')
    non_nans = (~np.isnan(x) & ~np.isnan(y))
    if not any(non_nans):
        return [], [], []

    x = np.array(x[non_nans], dtype=float)
    y = np.array(y[non_nans], dtype=float)

    H, xedges, yedges, _ = scst.binned_statistic_2d(
            x=x,
            y=y,
            values=[],
            statistic='count',
            bins=[N, N],
            range=bounds
    )
    xmean = (xedges[1::] + xedges[0:-1])/2.
    ymean = (yedges[1::] + yedges[0:-1])/2.
    XX, YY = np.meshgrid(xmean, ymean)

    XX = XX.flatten()
    YY = YY.flatten()
    H = H.T.flatten()
    xv = XX[H > 0]
    yv = YY[H > 0]

    # Derive alpha/transparency using a log scale
    min_alpha = 0.10
    max_alpha = 1.0
    H = H[H > 0]
    # H = H / np.max(H)
    H = np.log(H)/np.log(10)
    H = H / np.max(H)
    H[H > max_alpha] = max_alpha
    H[H < min_alpha] = min_alpha
    return xv, yv, H


def _plot_by_transparency_bins(ax, x, y, z, alpha_edges,
                               markersize=3, color='k',
                               plotlabel=None):

    dalpha = np.unique(np.diff(alpha_edges))[0]
    alpha_means = alpha_edges + dalpha / 2.
    z_bins = np.digitize(x=z, bins=alpha_edges)
    for ii in range(1, len(alpha_edges) + 1):
        alpha = alpha_means[ii-1]
        x_sub = x[z_bins == ii]
        y_sub = y[z_bins == ii]
        if ii == 1:
            plotlabel = plotlabel
        else:
            plotlabel = None

        ax.plot(x_sub, y_sub, '.', color=color,
                markersize=markersize,
                alpha=alpha, label=plotlabel)
    return ax


def _make_confirmation_plot(df, ti=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Check status flag of the dataframe
    is_ok = (df['status_%03d' % ti] == 1)

    # Show all data points
    x, y, H = _approximate_large_scatter_plot(
        x=df.loc[~is_ok, 'ws_%03d' % ti],
        y=df.loc[~is_ok, 'pow_%03d' % ti])
    ax.plot(x, y, '.', color='r', markersize=3)

    # Show the okay data points
    x, y, H = _approximate_large_scatter_plot(
        x=df.loc[is_ok, 'ws_%03d' % ti],
        y=df.loc[is_ok, 'pow_%03d' % ti])
    ax.plot(x, y, '.', color='k', markersize=3)

    ax.set_title('Turbine %03d' % ti)
    ax.legend(['Faulty data (%.1f %%)'
               % (100*np.sum(~is_ok)/len(is_ok)),
               'Filtered data (%.1f %%)'
               % (100*np.sum(is_ok)/len(is_ok))])
    ax.set_xlim([0., 30.])

    ax.set_ylabel('Power (kW)')
    ax.set_xlabel('Wind speed (m/s)')

    return ax


def plot_df_filtering(df, save_path_and_prefix=None, dpi=300):
    matplotlib.use('Agg')  # Non-GUI backend to speed up plots
    num_turbines = len([c for c in df.columns if
                        'status_' in c and len(c) == 10 and
                        '_all' not in c])
    dt = fsut.estimate_dt(df['time'])
    for ti in range(num_turbines):
        print('Producing confirmation plot for turbine %03d' % ti)
        if ti == 0:
            ax = _make_confirmation_plot(df=df, ti=ti)
        else:
            ax.clear()
            ax = _make_confirmation_plot(df=df, ti=ti, ax=ax)
        ax.set_title('Turbine %03d, dt = %.1f s' % (ti, dt.seconds))
        if save_path_and_prefix is not None:
            print('Saving confirmation plot for turbine %03d' % ti)
            fout = save_path_and_prefix + ('_%03d' % ti + '.png')
            plt.savefig(fout, dpi=dpi)


def plot_redzone(ax, x0, y0, dx, dy, text, fontsize=24, ii=0):
    plotcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ii = np.remainder(ii, len(plotcolors))
    clr = plotcolors[ii]

    r = ax.add_patch(
        patches.Rectangle((x0, y0), dx, dy,
                          linewidth=1, edgecolor=clr,
                          facecolor=clr, alpha=0.1))

    ax.add_artist(r)
    ax.annotate(text, (x0 + dx / 2., y0 + dy / 2.),
                color=clr, weight='bold',
                fontsize=fontsize, ha='center',
                va='center')

    return ax


def plot_filtering_distribution(N_list, label_list):
    fig, ax = plt.subplots(figsize=(10, 0.5))
    N_total = int(np.sum(N_list))
    N_list_nrm = [i/N_total for i in N_list]  # Normalize
    clrs = ['black', 'blue', 'orange', 'purple']
    edge_l = 0
    for ii in range(len(N_list)):
        edge_r = N_list_nrm[ii]
        plt.barh(0, edge_r, left=edge_l, color=clrs[ii])
        ax.text(edge_l + edge_r / 2., 0, label_list[ii] +
                ': %d (%.1f %%)' % (N_list[ii], N_list_nrm[ii]*100),
                ha='center', va='center', color='white')
        edge_l = edge_l + edge_r
    ax.set_xlim([0.0, 1.0])
    ax.set(yticklabels=[])
    return fig


# Function to convert list to ranges
def convert_list_to_ranges(list_in):
    def _ranges(i):
        for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]

    list_out = [range(i[0], i[1]+1) for i in (_ranges(list_in))]
    N = len(list_out)
    i = 0
    while i < N:
        v = list_out[i]
        if len(np.array(v)) == 1:
            list_out[i] = np.int64(v[0])
        if len(np.array(v)) == 2:
            list_out[i] = np.int64(v[0])
            list_out.insert(i+1, np.int64(v[1]))
            N = N + 1
            i = i + 1
        i = i + 1
    return list_out
