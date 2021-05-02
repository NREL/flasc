# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import datetime
from matplotlib import pyplot as plt
import numpy as np
import os as os
import pandas as pd
# from pandas.core.base import DataError
from scipy import optimize as opt
from scipy import stats as spst
from floris.utilities import wrap_360

from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as ftools
# from floris_scada_analysis import optimization as fopt
from floris_scada_analysis import scada_analysis as sca
from floris_scada_analysis import time_operations as tops


def printnow(text):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(now_time + ': ' + text)
    return True


class bias_estimation():
    def __init__(self, df, fi, test_turbines_subset, sliding_window_lb_ub,
                 df_ws_mapping_func, df_pow_ref_mapping_func,
                 df_upstream=None, df_fi_approx=None,  eo_ws_step=5.0,
                 eo_wd_step=2.0, df_ws_range=None, verbose=False):
        printnow('Initializing the bias_correction() class from floris_scada_analysis.')

        self.verbose = verbose

        # # Check format of dataframes based on first 100 rows
        # if df.shape[0] != df_fi.shape[0]:
        #     raise DataError('Please ensure df_fi and df have the same number of entries.')
        # if not (list(df.iloc[0:100]['time'])==list(df_fi.iloc[0:100]['time'])):
        #     raise DataError("The columns 'time' should be identical between df and df_fi.")
        # if not (list(df.iloc[0:100]['wd'])==list(df_fi.iloc[0:100]['wd'])):
        #     raise DataError("The columns 'wd' should be identical between df and df_fi.")
        # if not (list(df.iloc[0:100]['ws'])==list(df_fi.iloc[0:100]['ws'])):
        #     raise DataError("The columns 'ws' should be identical between df and df_fi.")

        df = df.reset_index(drop=('time' in df.columns))
        # df_fi = df_fi.reset_index(drop=('time' in df_fi.columns))

        # Import inputs
        self.fi = fi  # Floris object
        self.df = df  # Actual measurements
        if df_fi_approx is None:
            df_fi = df[['time', 'ws', 'wd']].copy()
            _, df_fi_approx = ftools.calc_floris_approx(
                df_fi, fi, wd_step=2.0, num_threads=40)
        self.df_fi_approx = df_fi_approx  # Save FLORIS solutions LUT
        self.df_subset = None # Subset of interest (actual)

        self.df_ws_mapping_func = df_ws_mapping_func
        self.df_pow_ref_mapping_func = df_pow_ref_mapping_func
        # self.df_fi = df_fi  # Floris predictions
        # self.df_fi_subset = None # Subset of interest (floris)

        self.turbine_list = range(len(fi.layout_x))
        self.test_turbines_subset = test_turbines_subset
        # self.ref_turbine_maxrange = ref_turbine_maxrange
        self.eo_wd_step = eo_wd_step
        self.eo_ws_step = eo_ws_step
        if df_ws_range is None:
            df_ws_range = (0., 50.)
        self.df_ws_range = df_ws_range

        self._reset_results()

        self.df_upstream = df_upstream
        self._map_ref_turbs_floris()

        printnow('  Initializing sliding window')
        time_array = list(pd.to_datetime(df.time))
        self.sw_time_array = time_array
        self.sw_current_time = time_array[0]
        self.sw_lb_ub = sliding_window_lb_ub
        self.sw_time_array_dt = tops.estimate_dt(time_array)
        printnow('  Estimated the data to be sampled at %.1f seconds.'
                 % (self.sw_time_array_dt.seconds))

        # Introduce variables that indicate index of time_array for sliding window
        self._sw_lower_idx = 0
        self._sw_upper_idx = -1

        # Update variables and update dataframe
        self.set_current_time(self.sw_current_time)

    def set_current_time(self, new_time):
        timestep = new_time - self.sw_current_time
        self.sw_current_time = new_time
        self._update_sw_indices(timestep)
        self._reset_results()
        self._update_df_subset()

    def _update_df_subset(self):
        printnow('Updated dataframes according to sliding window lock.')
        self.df_subset = self.df.loc[self._sw_lower_idx:self._sw_upper_idx].copy()
        # self.df_fi_subset = self.df_fi.loc[self._sw_lower_idx:self._sw_upper_idx].copy()

    def _reset_results(self):
        # Set-up outputs
        self.energy_ratios_scada = None
        self.energy_ratios_floris = None

        # Set default values for the optimization
        self.opt_wd_bias = np.nan  # Estimated bias in degrees
        self.opt_success = False

    def _update_sw_indices(self, timestep):
        current_time = self.sw_current_time
        time_array = self.sw_time_array
        dt = self.sw_time_array_dt
        sliding_window_lb_ub = self.sw_lb_ub

        # Create sliding window
        sliding_window_stepsize_didx = int(timestep / dt)
        sliding_window_width_didx = int((sliding_window_lb_ub[1]-sliding_window_lb_ub[0]) / dt)

        # Find corresponding index to time_array_60s
        window_time_min = current_time + sliding_window_lb_ub[0]
        window_time_max = current_time + sliding_window_lb_ub[1]

        # Do _min first
        if window_time_min <= time_array[0]:
            self._sw_lower_idx = 0  # boundary condition
        else:
            # We guess that the new window should be here
            self._sw_lower_idx += sliding_window_stepsize_didx + 2
            self._sw_lower_idx = int(np.max([self._sw_lower_idx, 0]))
            self._sw_lower_idx = int(np.min([self._sw_lower_idx, len(time_array)-1]))

            # Step back until we are actually right on the window
            while time_array[self._sw_lower_idx] > window_time_min:
                self._sw_lower_idx += -1
            self._sw_lower_idx += 1

        if window_time_max >= time_array[-1]:
            self._sw_upper_idx = len(time_array) - 1 # boundary condition
        else:
            # We guess that the new window should be here
            self._sw_upper_idx = self._sw_lower_idx + sliding_window_width_didx
            self._sw_upper_idx = np.min([self._sw_upper_idx, len(time_array)-1])
            # Step back until we are actually right on the window
            while time_array[self._sw_upper_idx] > window_time_max:
                self._sw_upper_idx += -1

        time_min = self.sw_time_array[self._sw_lower_idx]
        time_max = self.sw_time_array[self._sw_upper_idx]
        printnow('Sliding window locked onto time range (' + time_min.strftime('%Y-%m-%d %H:%M:%S') +
                 ', ' + time_max.strftime('%Y-%m-%d %H:%M:%S') + ') with current time (' +
                 current_time.strftime('%Y-%m-%d %H:%M:%S') + ').')

    # Determine which turbines are freestream for certain WD
    def _map_ref_turbs_floris(self):
        fi = self.fi
        df_upstream = self.df_upstream
        if df_upstream is None:
            printnow('  Determining upstream turbines per wind direction using FLORIS for wd_step = 5.0 deg.')
            df_upstream = ftools.get_upstream_turbs_floris(fi, wd_step=5.0)
            self.df_upstream = df_upstream

        self.upstream_turbs_wds = [
            [df_upstream.loc[i, 'wd_min'], df_upstream.loc[i, 'wd_max']]
            for i in range(df_upstream.shape[0])]
        self.upstream_turbs_ids = list(df_upstream.turbines)

    def _get_energy_ratios_allbins(self, wd_bias=0.0, N_btstrp=1,
                                   plot_iter_path=None):
        wd_step = self.eo_wd_step
        ws_step = self.eo_ws_step

        # Create dataframes: time-shifted df
        df_subset_cor = self.df_subset.copy()
        df_subset_cor['wd'] = wrap_360(df_subset_cor['wd'] - wd_bias)

        # Set columns 'ws' and 'pow_ref' for df_subset_cor
        df_subset_cor = self.df_ws_mapping_func(df_subset_cor)
        df_subset_cor = self.df_pow_ref_mapping_func(df_subset_cor)

        # Limit dataframe to ranges and remove NaN entries
        df_subset_cor = dfm.filter_df_by_ws(df_subset_cor, self.df_ws_range)
        df_subset_cor = df_subset_cor.dropna(subset=['time', 'wd',
                                                     'ws', 'pow_ref'])
        df_subset_cor  = df_subset_cor.reset_index(drop=True)

        # Get FLORIS predictions
        df_fi_subset = df_subset_cor[['time', 'wd', 'ws']].copy()
        df_fi_subset, _ = ftools.calc_floris_approx(df=df_subset_cor, fi=self.fi,
                                                    df_approx=self.df_fi_approx)
        df_fi_subset = self.df_pow_ref_mapping_func(df_fi_subset)

        # Initialize SCADA analysis class and add dataframes
        fsc = sca.scada_analysis(verbose=self.verbose)
        fsc.add_df(df_subset_cor, 'Measurement data')
        fsc.add_df(df_fi_subset, 'FLORIS predictions')

        test_turbines = self.test_turbines_subset
        self.energy_ratios_scada = [[] for _ in test_turbines]
        self.energy_ratios_floris = [[] for _ in test_turbines]

        for ii, ti in enumerate(test_turbines):
            printnow('  Determining energy ratio for wd_bias = ' +
                     '%.3f and test turbine = %03d...' % (wd_bias, ti))
            df_energyratio_scada_ti = pd.DataFrame()
            df_energyratio_floris_ti = pd.DataFrame()
            for wd_bin_i in range(len(self.upstream_turbs_wds)):
                wd_range = (self.upstream_turbs_wds[wd_bin_i][0],
                            self.upstream_turbs_wds[wd_bin_i][1])

                fsc.set_masks(wd_range=wd_range)
                fsc.get_energy_ratios(test_turbines=[ti], wd_step=wd_step,
                                      ws_step=ws_step, N=N_btstrp)

                # Save outputs as a merged dataframe
                out_scada = fsc.df_list[0]['er_results']
                out_floris = fsc.df_list[1]['er_results']
                df_energyratio_scada_ti = (
                    df_energyratio_scada_ti.append(
                        out_scada, ignore_index=True)
                )
                df_energyratio_floris_ti = (
                    df_energyratio_floris_ti.append(
                        out_floris, ignore_index=True)
                )

            self.energy_ratios_scada[ii] = df_energyratio_scada_ti
            self.energy_ratios_floris[ii] = df_energyratio_floris_ti

        # Debugging: show all possible options
        if plot_iter_path is not None:
            fp = os.path.join(plot_iter_path, 'bias%+.3f' % (wd_bias), 'energyratio')
            os.makedirs(os.path.basename(fp), exist_ok=True)
            self.plot_energy_ratios(save_path=fp, format='png')
            plt.close('all')

        return None

    def estimate_wd_bias(self, opt_search_range=(-180., 180.),
                         opt_search_dx=1.0, opt_finish=opt.fmin,
                         energy_ratio_N_btstrp=1,
                         plot_iter_path=None):

        printnow('Estimating the wind direction bias')
        def cost_fun(wd_bias):
            self._get_energy_ratios_allbins(wd_bias=wd_bias, N_btstrp=1,
                                            plot_iter_path=plot_iter_path)

            # Unpack variables
            # wd_arrays = [self.energy_ratios_scada[i]['wd_bin'] for i
            #             in range(len(self.energy_ratios_scada))]
            cost_array = np.full(len(self.energy_ratios_scada), np.nan)
            for ii in range(len(self.energy_ratios_scada)):
                y_scada = np.array(self.energy_ratios_scada[ii]['baseline'])
                y_floris = np.array(self.energy_ratios_floris[ii]['baseline'])
                ids = ~np.isnan(y_scada) & ~np.isnan(y_floris)
                r, p = spst.pearsonr(y_scada[ids], y_floris[ids])
                cost_array[ii] = -1. * r

            cost = np.mean(cost_array)
            return cost

        if opt_finish is not None:
            opt_finish = (
                lambda func, x0, args=(): opt.fmin(func, x0, args,
                                                   full_output=True,
                                                   xtol=0.1, disp=True)
            )

        dran = opt_search_range[1]-opt_search_range[0]
        x_opt, J_opt, x, J = opt.brute(
            func=cost_fun,
            ranges=[opt_search_range],
            Ns=int(np.ceil(dran/opt_search_dx) + 1),
            full_output=True,
            disp=True,
            finish=opt_finish)

        wd_bias = x_opt
        self.opt_wd_bias = wd_bias
        self.opt_cost = J_opt
        self.opt_wd_grid = x
        self.opt_wd_cost = J

        # End with optimal results and bootstrapping
        self._get_energy_ratios_allbins(
            wd_bias=x_opt, N_btstrp=energy_ratio_N_btstrp)

        return x_opt, J_opt

    def plot_energy_ratios(self, save_path=None, format='png'):
        import matplotlib.pyplot as plt

        # Unpack variables
        wd_arrays = [self.energy_ratios_scada[i]['wd_bin'] for i
                     in range(len(self.energy_ratios_scada))]
        er_result_scada = [self.energy_ratios_scada[i]['baseline'] for i
                           in range(len(self.energy_ratios_scada))]
        er_result_scada_l = [self.energy_ratios_scada[i]['baseline_l'] for i
                           in range(len(self.energy_ratios_scada))]
        er_result_scada_u = [self.energy_ratios_scada[i]['baseline_u'] for i
                           in range(len(self.energy_ratios_scada))]
        er_result_floris = [self.energy_ratios_floris[i]['baseline'] for i
                            in range(len(self.energy_ratios_floris))]
        er_result_floris_l = [self.energy_ratios_floris[i]['baseline_l'] for i
                            in range(len(self.energy_ratios_floris))]
        er_result_floris_u = [self.energy_ratios_floris[i]['baseline_u'] for i
                            in range(len(self.energy_ratios_floris))]

        printnow('Plotting results for last evaluated case')
        printnow('  Wind direction bias: %.1f deg' % (self.opt_wd_bias))
        for ii in range(len(self.test_turbines_subset)):
            ti = self.test_turbines_subset[ii]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(wd_arrays[ii], er_result_scada[ii], color='k', label='SCADA data')
            ax.fill_between(wd_arrays[ii], er_result_scada_l[ii], er_result_scada_u[ii], alpha=0.15)
            # if not np.isnan(self.opt_wd_bias):
            #     x_corrected = wrap_360(wd_arrays[ii] - self.opt_wd_bias)
            #     y_corrected = np.array(er_result_scada[ii])[np.argsort(x_corrected)]
            #     y_corrected_l = np.array(er_result_scada_l[ii])[np.argsort(x_corrected)]
            #     y_corrected_u = np.array(er_result_scada_u[ii])[np.argsort(x_corrected)]
            #     x_corrected = np.sort(x_corrected)
            #     ax.plot(x_corrected, y_corrected, color='blue', label='SCADA data (corrected)')
            #     ax.fill_between(x_corrected, y_corrected_l, y_corrected_u, alpha=0.15)
            ax.plot(wd_arrays[ii], er_result_floris[ii], ls='--', color='orange', label='FLORIS')
            ax.fill_between(wd_arrays[ii], er_result_floris_l[ii], er_result_floris_u[ii], alpha=0.15)
            plt.title('Turbine %d. Current time: %s' % (ti, str(self.sw_current_time)))
            plt.ylabel('Energy ratio (-)')
            plt.xlabel('Wind direction (deg)')
            plt.grid(b=True, which='major', axis='both', color='gray')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', axis='both', color='lightgray')
            plt.legend()

            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path + '_%03d.%s' % (ti, format))
