# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os
import scipy.stats as scst

from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import sqldatabase_management as sqldbm
from floris_scada_analysis import time_operations as fsato

from operational_analysis.toolkits import filters


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

    x = x[non_nans]
    y = y[non_nans]

    H, xedges, yedges, _ = (
        scst.binned_statistic_2d(x=x, y=y, values=None,
                                 statistic='count', bins=[N, N],
                                 range=bounds))
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
    ax.legend(['Faulty data', 'Filtered data'])
    ax.set_xlim([0., 30.])

    ax.set_ylabel('Power (kW)')
    ax.set_xlabel('Wind speed (m/s)')

    return ax


def plot_df_filtering(df, save_path_and_prefix=None, dpi=300):
    matplotlib.use('Agg')  # Non-GUI backend to speed up plots
    num_turbines = len([c for c in df.columns if
                        'status_' in c and len(c) == 10 and
                        '_all' not in c])
    dt = fsato.estimate_dt(df['time'])
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


def plot_filtering_distribution(N, N_oow, N_oowsdev):
    fig, ax = plt.subplots(figsize=(10, 0.5))
    N_clean = N-N_oow-N_oowsdev
    plt.barh(0, N_clean/N, left=0, color='black')
    plt.barh(0, N_oow/N, left=N_clean/N, color='orange')
    plt.barh(0, N_oowsdev/N, left=(N-N_oowsdev)/N, color='purple')

    ax.text(N_clean/(2*N), 0, '%d valid datapoints (%.1f %%)'
            % (N_clean, N_clean*100/N), ha='center', va='center',
            color='white')
    ax.text(N_clean/N + N_oow/(2*N), 0, '%.1f%%' % (N_oow*100/N),
            ha='center', va='center', color='w')
    ax.text(N_clean/N + N_oow/N + N_oowsdev/(2*N), 0,
            '%.1f%%' % (N_oowsdev*100/N), ha='center',
            va='center', color='w')
    ax.axis('off')

    return fig


class ws_pw_curve_filtering():
    def __init__(self, df, single_turbine_mode=False):
        # Check format of df
        if 'self_status_000' not in df.columns:
            print('No self_status flags found. Assuming all are 1.')
            num_turbines = dfm.get_num_turbines(df)
            for ti in range(num_turbines):
                df['self_status_%03d' % ti] = int(1)

        # Assign dataframe to self
        self.set_df(df)

        # Set self.num_turbines to 1 or to all turbines
        self.set_turbine_mode(single_turbine_mode)

        # Setup windows
        self.window_list = []

        # Setup empty dataframe for esti
        self.pw_curve_df = None

        # Derive information from turbine 0 in dataset
        for ti in [0]:
            est_ratedpw = np.median(
                df.loc[df['ws_%03d' % ti] > 15., 'pow_%03d' % (ti)])
            if est_ratedpw < 20.0:
                est_ratedpw = np.round(est_ratedpw, 1)  # MW
            elif est_ratedpw < 20.0e3:
                est_ratedpw = np.round(est_ratedpw/1e3, 1)*1e3  # kW
            else:
                est_ratedpw = np.round(est_ratedpw/1e6, 1)*1e6  # W
        est_ratedpw = float(est_ratedpw)
        print('Estimated rated power of turbines in this dataset' +
              ' to be %.1f' % est_ratedpw)

        # Derive default settings
        default_pow_step = 50
        default_ws_dev = 2.0
        default_max_pow_bin = 0.95 * est_ratedpw
        default_w0_ws = (0.0, 15.0)
        default_w0_pw = (0.0, 0.95 * est_ratedpw)
        default_w1_ws = (0.0, 25.0)
        default_w1_pw = (0.0, 1.04 * est_ratedpw)

        # Setup windows and binning properties
        self.window_add(default_w0_ws, default_w0_pw, axis=0)
        self.window_add(default_w1_ws, default_w1_pw, axis=1)
        self.set_binning_properties(pow_step=default_pow_step,
                                    ws_dev=default_ws_dev,
                                    max_pow_bin=default_max_pow_bin,
                                   )

    def set_df(self, df):
        # Make sure dataframe index is uniformly ascending and save
        self.df = df.reset_index(drop=('time' in df.columns))
        self.dt = fsato.estimate_dt(self.df['time'])

    def set_turbine_mode(self, single_turbine_mode):
        if single_turbine_mode:
            self.num_turbines = 1
        else:
            self.num_turbines = dfm.get_num_turbines(self.df)

    def set_binning_properties(self, pow_step=None,
                               ws_dev=None, max_pow_bin=None):
        if pow_step is not None:
            self.pow_step = pow_step
        if ws_dev is not None:
            self.ws_dev = ws_dev
        if max_pow_bin is not None:
            self.max_pow_bin = max_pow_bin

    def window_add(self, ws_range, pow_range, axis=0):
        # axis=0 means limiting values lower/higher than pow
        # axis=1 means limiting values lower/higher than ws
        idx = len(self.window_list)
        new_entry = {'idx': idx,
                     'ws_range': ws_range,
                     'pow_range': pow_range,
                     'axis': axis}
        self.window_list.append(new_entry)

    def window_remove(self, i):
        self.window_list.pop(i)
        # Update indices
        for i in range(len(self.window_list)):
            self.window_list[i]['idx'] = i

    def window_remove_all(self):
        self.window_list = []

    def window_print_all(self):
        for i in range(len(self.window_list)):
            window = self.window_list[i]
            for k in window.keys():
                print("window_list[%d][%s] = " % (i, k),
                      self.window_list[i][k])

    def apply_filters(self):
        self.df_out_of_windows = []
        self.df_out_of_ws_dev = []
        for ti in range(self.num_turbines):
            print(' ')
            print('Applying window filters to the df for turbine %d' % ti)

            # Filter by self flag
            df = self.df.copy()
            is_ok = (df['self_status_%03d' % ti].values == 1)
            df = df.loc[is_ok]

            out_of_window_ids = np.zeros(df.shape[0])
            for window in self.window_list:
                idx = window['idx']
                ws_range = window['ws_range']
                pow_range = window['pow_range']
                axis = window['axis']
                if axis == 0:
                    ii_out_of_window = (
                        filters.window_range_flag(df['pow_%03d' % ti],
                                                  pow_range[0],
                                                  pow_range[1],
                                                  df['ws_%03d' % ti],
                                                  ws_range[0],
                                                  ws_range[1])
                        )
                else:
                    ii_out_of_window = (
                        filters.window_range_flag(df['ws_%03d' % ti],
                                                  ws_range[0],
                                                  ws_range[1],
                                                  df['pow_%03d' % ti],
                                                  pow_range[0],
                                                  pow_range[1])
                        )

                # Merge findings from all windows
                out_of_window_ids[ii_out_of_window] = int(1)
                print('  Removed %d outliers using window[%d].'
                    % (int(sum(ii_out_of_window)), idx))

            print('Removed a total of %d outliers using the %d windows.'
                  % (int(sum(out_of_window_ids)), len(self.window_list)))
            self.df_out_of_windows.append([bool(i) for i in out_of_window_ids])

            # Filter by standard deviation for the reduced dataset
            df_ok = df[[not bool(i) for i in out_of_window_ids]]
            out_of_dev_series = filters.bin_filter(
                df_ok['pow_%03d' % ti], df_ok['ws_%03d' % ti],
                self.pow_step, self.ws_dev, 'median', 20.,
                self.max_pow_bin, 'scalar', 'all')
            out_of_dev_indices = df_ok.index[np.where(out_of_dev_series)[0]]
            df_out_of_ws_dev = np.zeros(df.shape[0])
            df_out_of_ws_dev[out_of_dev_indices] = 1
            self.df_out_of_ws_dev.append([bool(i) for i in df_out_of_ws_dev])
            print('Removed %d outliers using WS standard deviation filtering.'
                  %(int(sum(df_out_of_ws_dev))))

            # Add a status flag for this turbine
            self.df['status_%03d' % ti] = 1
            self.df.loc[self.df_out_of_ws_dev[-1], 'status_%03d' % ti] = 0
            self.df.loc[self.df_out_of_windows[-1], 'status_%03d' % ti] = 0

        # Add a status_all column
        status_cols = [('status_%03d' % ti) for ti in range(self.num_turbines)]
        self.df['status_all'] = self.df[status_cols].min(axis=1)
        self.pw_curve_df = None # Reset estimated power curve
        return self.df

    def extract_power_curve(self, ws_bins=np.arange(0.0, 25.5, 0.5)):
        ws_max = np.max(ws_bins)
        ws_min = np.min(ws_bins)
        pw_curve_df = pd.DataFrame({
            'ws': (ws_bins[1::]+ws_bins[0:-1])/2,
            'ws_min': ws_bins[0:-1], 'ws_max': ws_bins[1::]})

        for ti in range(self.num_turbines):
            ws = self.df['ws_%03d' % ti]
            pow = self.df['pow_%03d' % ti]
            status = self.df['status_%03d' % ti]
            clean_ids = ((status == 1) & (ws > ws_min) & (ws < ws_max))
            ws_clean = ws[clean_ids]
            pw_clean = pow[clean_ids]

            # bin_array = np.digitize(ws_clean, ws_bins_l, right=False)
            bin_array = np.searchsorted(ws_bins, ws_clean, side='left')
            bin_array = bin_array - 1  # 0 -> 1st bin, rather than before bin
            pow_bins = [np.mean(pw_clean[bin_array==i])
                        for i in range(pw_curve_df.shape[0])]

            # Write outputs to the dataframe
            pw_curve_df['pow_%03d' % ti] = pow_bins
            self.pw_curve_df = pw_curve_df

        return pw_curve_df

    def save_df(self, fout):
        status_cols = [('status_%03d' % ti) for ti in range(self.num_turbines)]

        # Remove df entries with all status == 0
        df = self.df.copy()
        all_bad = (df[status_cols].max(axis=1) == 0)
        df = df[[not bool(i) for i in all_bad]]
        df = df.reset_index(drop=('time' in df.columns))
        df.to_feather(fout)

    def plot(self, pretty=False, draw_windows=True,
             confirm_plot=True,  save_path=None,
             fig_format='png', dpi=300):
        if pretty:
            fig_list = (
                self.plot_pretty(
                    draw_windows=draw_windows,
                    confirm_plot=confirm_plot,
                    save_path=save_path,
                    fig_format=fig_format, dpi=dpi)
            )
        else:
            fig_list = (
                self.plot_fast(
                    draw_windows=draw_windows,
                    confirm_plot=confirm_plot,
                    save_path=save_path,
                    fig_format=fig_format, dpi=dpi)
            )
        return fig_list

    def plot_fast(self, draw_windows=True, confirm_plot=True,
             save_path=None, fig_format='png', dpi=300):
        df = self.df

        fig_list = []
        for ti in range(self.num_turbines):
            print('Generating ws-power plot for turbine %03d' % ti)
            if confirm_plot:
                fig, ax = plt.subplots(1, 2, figsize=(28, 5))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(14, 5))
                ax = [ax]
            fig_list.append(fig)

            # Discretization variables
            xmin = np.min(df['ws_%03d' % ti])
            xmax = np.max(df['ws_%03d' % ti])
            ymin = np.min(df['pow_%03d' % ti])
            ymax = np.max(df['pow_%03d' % ti])
            bounds = [[xmin, xmax], [ymin, ymax]]
            dalpha = 0.05
            alpha_edges = np.arange(0.0, 1.0, dalpha)

            # Show the acceptable points
            oowsdev = self.df_out_of_ws_dev[ti]
            oow = self.df_out_of_windows[ti]
            good_ids = [not(a) and not(b) for a, b in zip(oow, oowsdev)]
            x = df.loc[good_ids, 'ws_%03d' % ti]
            y = df.loc[good_ids, 'pow_%03d' % ti]
            x_approx, y_approx, z = _approximate_large_scatter_plot(x, y, bounds=bounds)
            ax[0] = _plot_by_transparency_bins(ax=ax[0], x=x_approx,
                                               y=y_approx, z=z,
                                               alpha_edges=alpha_edges,
                                               markersize=3, color='k',
                                               plotlabel='Filtered data')
            ax[0].set_title('Turbine %03d, 60 s sampled data' % ti)

            # Show the points self-screened
            alpha = 0.80
            self_not_ok = [not bool(i) for i in df['self_status_%03d' % ti]]
            x = df.loc[self_not_ok, 'ws_%03d' % ti]
            y = df.loc[self_not_ok,'pow_%03d' % ti]
            x_approx, y_approx, z = _approximate_large_scatter_plot(x, y, bounds=bounds)
            ax[0].plot(x_approx, y_approx, '.', markerfacecolor='r',
                       markersize=2, alpha=alpha, label='Self-flagged data')

            # Show the points screened out of window
            alpha = 0.80
            x = df.loc[oow, 'ws_%03d' % ti]
            y = df.loc[oow, 'pow_%03d' % ti]
            x_approx, y_approx, z = _approximate_large_scatter_plot(x, y, bounds=bounds)
            ax[0].plot(x_approx, y_approx, '.', color='orange',
                       markersize=5, alpha=alpha, label='Window outliers')

            # Show the points screened using ws_dev
            alpha = 0.80
            x = df.loc[oowsdev, 'ws_%03d' % ti]
            y = df.loc[oowsdev, 'pow_%03d' % ti]
            x_approx, y_approx, z = _approximate_large_scatter_plot(x, y, bounds=bounds)
            ax[0].plot(x_approx, y_approx, '.', color='purple',
                       markersize=5, alpha=alpha, label='WS deviation outliers')

            # Show the approximated power curve, if calculated
            if self.pw_curve_df is not None:
                ax[0].plot(self.pw_curve_df['ws'],
                           self.pw_curve_df['pow_%03d' % ti], '--',
                           label='Approximate power curve')

            if draw_windows:
                xlim = (0., 30.)
                ylim = ax[0].get_ylim()
                for window in self.window_list:
                    ws_range = window['ws_range']
                    pow_range = window['pow_range']
                    axis = window['axis']
                    idx = window['idx']

                    if axis == 0:
                        # Filtered region left of curve
                        plot_redzone(ax[0], xlim[0], pow_range[0],
                                     ws_range[0] - xlim[0],
                                     pow_range[1] - pow_range[0],
                                     '%d' % idx, ii=idx)
                        # Filtered region right of curve
                        plot_redzone(ax[0], ws_range[1], pow_range[0],
                                     xlim[1] - ws_range[1],
                                     pow_range[1] - pow_range[0],
                                     '%d' % idx, ii=idx)
                    else:
                        # Filtered region above curve
                        plot_redzone(ax[0], ws_range[0], pow_range[1],
                                     ws_range[1] - ws_range[0],
                                     ylim[1] - pow_range[1],
                                     '%d' % idx, ii=idx)
                        # Filtered region below curve
                        plot_redzone(ax[0], ws_range[0], ylim[0],
                                     ws_range[1] - ws_range[0],
                                     pow_range[0] - ylim[0],
                                     '%d' % idx, ii=idx)
                    # ax[0].add_patch(rect)

            ax[0].set_xlim(xlim)
            ax[0].set_ylim(ylim)
            
            ax[0].set_ylabel('Power (kW)')
            ax[0].set_xlabel('Wind speed (m/s)')

            ax[0].legend()
            if confirm_plot:
                _make_confirmation_plot(df, ti=ti, ax=ax[1])

            if save_path is not None:
                plt.savefig(save_path + '/%03d.' % ti + fig_format, dpi=dpi)

        return fig_list

    def plot_pretty(self, draw_windows=True, confirm_plot=True,
        save_path=None, fig_format='png', dpi=300):
        df = self.df

        fig_list = []
        for ti in range(self.num_turbines):
            print('Generating ws-power plot for turbine %03d' % ti)
            if confirm_plot:
                fig, ax = plt.subplots(1, 2, figsize=(28, 5))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(14, 5))
                ax = [ax]

            fig_list.append(fig)

            # Show the acceptable points
            alpha = 0.10
            oowsdev = self.df_out_of_ws_dev[ti]
            oow = self.df_out_of_windows[ti]
            good_ids = [not(a) and not(b) for a, b in zip(oow, oowsdev)]
            x = df.loc[good_ids, 'ws_%03d' % ti]
            y = df.loc[good_ids, 'pow_%03d' % ti]
            ax[0].plot(x, y, '.', color='k', markersize=3, alpha=alpha)
            ax[0].set_title('Turbine %03d, 60 s sampled data' % ti)

            # Show the points self-screened
            alpha = 0.30
            self_not_ok = [not bool(i) for i in df['self_status_%03d' % ti]]
            ax[0].plot(df.loc[self_not_ok, 'ws_%03d' % ti],
                    df.loc[self_not_ok,'pow_%03d' % ti],
                    '.', markerfacecolor='r', markersize=2, alpha=alpha)

            # Show the points screened out of window
            alpha = 0.30
            ax[0].plot(df.loc[oow, 'ws_%03d' % ti],
                    df.loc[oow, 'pow_%03d' % ti],
                    '.', color='orange', markersize=5, alpha=alpha)

            # Show the points screened using ws_dev
            alpha = 0.30
            ax[0].plot(df.loc[oowsdev, 'ws_%03d' % ti],
                    df.loc[oowsdev, 'pow_%03d' % ti],
                    '.', color='purple', markersize=5, alpha=alpha)

            # Show the approximated power curve, if calculated
            if self.pw_curve_df is not None:
                ax[0].plot(self.pw_curve_df['ws'], self.pw_curve_df['pow_%03d' % ti], '--')

            if draw_windows:
                xlim = (0., 30.)  #ax[0].get_xlim()
                ylim = ax[0].get_ylim()
                for window in self.window_list:
                    ws_range = window['ws_range']
                    pow_range = window['pow_range']
                    axis = window['axis']
                    idx = window['idx']

                    if axis == 0:
                        # Filtered region left of curve
                        plot_redzone(ax[0], xlim[0], pow_range[0],
                                     ws_range[0] - xlim[0],
                                     pow_range[1] - pow_range[0],
                                     '%d' % idx, ii=idx)
                        # Filtered region right of curve
                        plot_redzone(ax[0], ws_range[1], pow_range[0],
                                     xlim[1] - ws_range[1],
                                     pow_range[1] - pow_range[0],
                                     '%d' % idx, ii=idx)
                    else:
                        # Filtered region above curve
                        plot_redzone(ax[0], ws_range[0], pow_range[1],
                                     ws_range[1] - ws_range[0],
                                     ylim[1] - pow_range[1],
                                     '%d' % idx, ii=idx)
                        # Filtered region below curve
                        plot_redzone(ax[0], ws_range[0], ylim[0],
                                     ws_range[1] - ws_range[0],
                                     pow_range[0] - ylim[0],
                                     '%d' % idx, ii=idx)
                    # ax[0].add_patch(rect)

            ax[0].set_xlim(xlim)
            ax[0].set_ylim(ylim)
            
            ax[0].set_ylabel('Power (kW)')
            ax[0].set_xlabel('Wind speed (m/s)')

            if self.pw_curve_df is None:
                ax[0].legend(['Filtered data',
                              'Self-flagged data',
                              'Window outliers',
                              'WS deviation outliers'])
            else:
                ax[0].legend(['Filtered data',
                              'Self-flagged data',
                              'Window outliers',
                              'WS deviation outliers',
                              'Approximate power curve'])

            if confirm_plot:
                _make_confirmation_plot(df, ti=ti, ax=ax[1])

            if save_path is not None:
                plt.savefig(save_path + '/%03d.' % ti + fig_format, dpi=dpi)

        return fig_list

    def apply_filtering_to_other_df(self, df_target, threshold=.999, fout=None):

        if df_target.shape[0] < 2:
            # Too few entries: just assume status is bad
            status_cols = [('status_%03d' % ti) for
                           ti in range(self.num_turbines)]
            df_target[status_cols] = int(0)
            return df_target

        time_array_target = df_target['time']
        dt_target = fsato.estimate_dt(time_array_target)
        if dt_target >= 2.0 * self.dt:
            stws = [[t - dt_target, t] for t in time_array_target]
            time_map = fsato.find_window_in_time_array(
                time_array_src=self.df['time'],
                seek_time_windows=stws)

            for ti in range(self.num_turbines): # Base decision on threshold (-) of data
                print('Applying filtering to target_df with dt = %.1f s, turbine %03d.' % (dt_target.seconds, ti))
                # any_bad_ids = [(np.min(self.df.loc[ids, 'status_%03d' % ti])) for ids in time_map]
                bad_ids = [(np.mean(self.df.loc[ids, 'status_%03d' % ti]) < threshold) for ids in time_map]
                df_target['status_%03d' % ti] = int(1)
                df_target.loc[bad_ids, 'status_%03d' % ti] = int(0)
                print('  Mapping yields %d entries (%.2f %%) flagged as bad for ti = %d.'
                      % (np.sum(bad_ids), 100. * np.sum(bad_ids) / df_target.shape[0], ti))
        else:
            for ti in range(self.num_turbines):
                print('Applying filtering to target_df with dt = %.1f s, turbine %03d.' % (dt_target.seconds, ti))
                status_bad = self.df[('status_%03d' % ti)] == 0
                time_array_src_bad = self.df.loc[status_bad, 'time']
                stws = [[t - self.dt, t] for t in time_array_src_bad]
                bad_ids = fsato.find_window_in_time_array(
                    time_array_src=time_array_target,
                    seek_time_windows=stws)

                df_target['status_%03d' % ti] = int(1)
                if bad_ids is not None:
                    bad_ids = np.concatenate(bad_ids)
                    df_target.loc[bad_ids, 'status_%03d' % ti] = int(0)
                nbad = np.sum(1-df_target['status_%03d' % ti])
                print('  Mapping yields %d entries (%d %%) flagged as bad for ti = %d.'
                      % (nbad, 100. * nbad / df_target.shape[0], ti))

        status_cols = [('status_%03d' % ti) for ti in range(self.num_turbines)]
        df_target['status_all'] = df_target[status_cols].min(axis=1)

        # Remove df entries with all status == 0
        all_bad = (df_target[status_cols].max(axis=1) == 0)
        df_target = df_target[[not bool(i) for i in all_bad]]

        if fout is not None:
            print('Saving dataframe to ', fout)
            df_target = df_target.reset_index(
                drop=('time' in df_target.columns))
            df_target.to_feather(fout)

        return df_target


# Example on how to use this class for data filtering
if __name__ == '__main__':
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, 'data/01_structured_data')
    df_60s_filelist = sqldbm.browse_datafiles(data_path=data_path,
                                              scada_table='scada_data_60s')
    df_60s = sqldbm.batch_load_and_concat_dfs(df_filelist=df_60s_filelist)

    # Setup a wind-speed power curve filtering class
    ws_pow_filtering = ws_pw_curve_filtering(df=df_60s,
                                             single_turbine_mode=False)

    # Filter data using default settings
    ws_pow_filtering.apply_filters()

    # Plot and save data for current dataframe
    save_path = root_path + '/data/02_wspow_filtered_data/'
    ws_pow_filtering.plot(draw_windows=True,
                          confirm_plot=True,
                          save_path=save_path)
    plt.close('all')
    ws_pow_filtering.df.to_feather(root_path +
                                   '/data/02_wspow' +
                                   '_filtered_data' +
                                   '/scada_data_60' +
                                   's.ftr')

    # Apply same filters on down/upsampled data
    df_1s_filelist = sqldbm.browse_datafiles(
        data_path=data_path, scada_table='scada_data_1s')
    for df_fn in df_1s_filelist:
        print('Processing filtering for file %s' % df_fn)
        df_1s = pd.read_feather(df_fn)
        threshold = 0.99  # At least this ratio of data should be status == 1
        save_path = (root_path + '/data/02_wspow_filtered_data/'
                     + os.path.basename(df_fn))
        df_1s = ws_pow_filtering.apply_filtering_to_other_df(
            df_target=df_1s, threshold=threshold, fout=save_path)
        fig_save_path_and_prefix = (root_path + '/data/02_wspow_filtered_data/'
                                    + os.path.basename(df_fn) + '_plot')
        plot_df_filtering(df_1s, save_path_and_prefix=fig_save_path_and_prefix)
        plt.close('all')
