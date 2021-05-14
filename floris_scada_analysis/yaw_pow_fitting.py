# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import optimize as opt

from floris.utilities import wrap_180

from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import utilities as fsut


class yaw_pow_fitting():
    def __init__(self, df, df_upstream, ws_range=(6., 10.), turbine_list='all'):
        print('Initializing yaw power curve filtering object.')
        # Assign dataframes to self
        self.df_upstream = df_upstream
        self.set_df(df, ws_range)

        # Set turbines specified by user
        self.set_turbine_mode(turbine_list)

    def set_df(self, df, ws_range=(6., 10.)):
        if 'vane_000' not in df.columns:
            raise KeyError('vane_000 not found in dataset.')

        # Get true total number of turbines
        self.num_turbines_all = fsut.get_num_turbines(df)
        self.full_turbine_list = range(self.num_turbines_all)

        # Cut down dataframe by WS
        df = dfm.set_ws_by_upstream_turbines(df, self.df_upstream)
        df = df[df['ws'] > ws_range[0]]
        df = df[df['ws'] < ws_range[1]]
        df = df.reset_index(drop=True)

        # Set df using only the relevant columns
        rlvnt_cols = ['wd']
        rlvnt_cols.extend(['pow_%03d' % ti for ti in range(self.num_turbines_all)])
        rlvnt_cols.extend(['vane_%03d' % ti for ti in range(self.num_turbines_all)])
        self.df = df[rlvnt_cols]

        # Reset output variables
        self.bins_x_list = [[] for _ in range(self.num_turbines_all)]
        self.bins_y_list = [[] for _ in range(self.num_turbines_all)]
        self.bins_N_list = [[] for _ in range(self.num_turbines_all)]

        self.x_opt_list = [(None, None) for _ in range(self.num_turbines_all)]
        self.bins_y_opt_list = [[] for _ in range(self.num_turbines_all)]

    def set_turbine_mode(self, turbine_list):
        if isinstance(turbine_list, str):
            if turbine_list == 'all':
                num_turbines = fsut.get_num_turbines(self.df)
                turbine_list = range(num_turbines)
            else:
                raise KeyError('Invalid turbine_list specified.')

        self.turbine_list = turbine_list
        self.num_turbines = len(turbine_list)

    def calculate_curves(self, vane_bounds=(-15., 15.), dv=1.0, Pmin=10.0):
        df_upstream = self.df_upstream
        turbine_list = self.turbine_list

        for ti in turbine_list:
            print('Determining yaw-power curve for turbine %03d...' % ti)

            print('  Retrieving relevant dataframe subset...')
            rel_cols = ['wd', 'vane_%03d' % ti]
            rel_cols.extend(['pow_%03d' % ti for ti in self.full_turbine_list])
            df = self.df[rel_cols].copy()

            # Filter by upstream conditions
            df_upstr_ti = df_upstream[[ti in tl for tl in df_upstream['turbines']]]
            df_upstr_ti = df_upstr_ti.reset_index(drop=True)
            in_range = [False for _ in range(df.shape[0])]
            for i in range(df_upstr_ti.shape[0]):
                wd_min = df_upstr_ti.loc[i, 'wd_min']
                wd_max = df_upstr_ti.loc[i, 'wd_max']
                in_range = in_range | ((df['wd'] >= wd_min) & (df['wd'] <= wd_max))
            df = df.loc[in_range]

            # Get reference power signals
            print('  Cutting down dataframe by minimum reference power')
            df = dfm.set_pow_ref_by_upstream_turbines(
                df, df_upstream, exclude_turbs=[ti])
            df = df[df['pow_ref'] > Pmin]

            # Define vane and (normalized) power measurements
            vane = wrap_180(np.array(df['vane_%03d' % ti]))

            # Filter for viable conditions
            ids_good = ((vane >= vane_bounds[0]) &
                        (vane <= vane_bounds[1]) &
                        (df['pow_%03d' % ti] > Pmin))
            vane = vane[ids_good]
            Pnorm = df.loc[ids_good, 'pow_%03d' % ti] / df.loc[ids_good, 'pow_ref']
            print('  Number of useful datapoints: %d.' % len(vane))

            # Bin data
            print('  Binning data...')
            bins_x = np.arange(vane_bounds[0], vane_bounds[1], dv)
            bins_y = np.zeros_like(bins_x)
            bins_N = np.zeros_like(bins_x)

            for ii, edge_x_l in enumerate(bins_x):
                edge_x_r = edge_x_l + dv
                yi = Pnorm[(vane >= edge_x_l) & (vane < edge_x_r)]
                bins_N[ii] = yi.shape[0]
                bins_y[ii] = np.nanmean(yi)
            bins_y = np.array(bins_y) / np.nanmax(bins_y[bins_N/np.max(bins_N) > 0.10])  # Normalize to 1

            self.bins_x_list[ti] = bins_x
            self.bins_y_list[ti] = bins_y
            self.bins_N_list[ti] = bins_N

    def estimate_cos_pp_fit(self, opt_bias_range=(-15., 15.),
                            opt_pp_range=(1.0, 10.0), opt_Ns=100):

        for ti in self.turbine_list:
            bins_x = self.bins_x_list[ti]
            bins_y = self.bins_y_list[ti]
            bins_N = self.bins_N_list[ti]

            if len(bins_x) <= 0:
                raise ValueError('Please calculate curves using ' +
                                 '.calculate_curves() before ' +
                                 'estimating a fit for turbine %03d.' % ti)

            # Define an approximating function
            def approx_func(x):
                y = np.cos((bins_x-x[0]) * np.pi / 180.)**x[1]
                return y

            # Define a cost function
            def cost(x):
                # x[0] is the x offset, x[1] is the cos coefficient
                y_fit = approx_func(x)
                J_fit = np.multiply(bins_N, (y_fit - bins_y)**2.)
                J_sum = np.nanmean(J_fit)
                return J_sum

            print('Fitting a cos(x-x0)^pp curve to the data of turbine %03d...' % ti)
            x_opt, J_opt, x, J = opt.brute(
                func=cost,
                ranges=(opt_bias_range, opt_pp_range),
                Ns=opt_Ns,
                finish=opt.fmin,
                full_output=True,
                disp=True
                )
            print('Turbine% 03d, x_opt: ' % ti, x_opt)
            y_opt = approx_func(x_opt)

            self.x_opt_list[ti] = x_opt
            self.bins_y_opt_list[ti] = y_opt

    def plot(self, save_path=None, fig_format='png', fig_dpi=250):
        for ti in self.turbine_list:
            bins_x = self.bins_x_list[ti]
            bins_y = self.bins_y_list[ti]
            bins_N = self.bins_N_list[ti]

            x_opt = self.x_opt_list[ti]
            y_opt = self.bins_y_opt_list[ti]

            if len(bins_x) <= 0:
                raise ValueError('Please calculate curves using ' +
                                 '.calculate_curves() before ' +
                                 'plotting for turbine %03d.' % ti)

            # Plot bins and averaged curve
            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(bins_x, bins_y, 'o-', label='Data')
            if len(y_opt) > 0:
                ax[0].plot(bins_x, y_opt, '--',
                        label=('Fit (x0=%.3f, x1=%.3f)'
                               % (x_opt[0], x_opt[1])))
            ax[0].set_xlabel('Vane measurement (deg)')
            ax[0].set_ylabel('Relative power production (-)')
            ax[0].set_title('Turbine %03d' % ti)
            ax[0].grid('minor')
            ax[0].legend()

            ax[1].bar(bins_x, bins_N)
            ax[1].set_xlabel('Vane measurement (deg)')
            ax[1].set_ylabel('Number of data points (-)')

            fig.tight_layout()
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(save_path + '/yawpowcurve_%03d.%s'
                            % (ti, fig_format), dpi=fig_dpi)
