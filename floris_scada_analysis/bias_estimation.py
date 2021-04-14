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
import numpy as np
import pandas as pd
from pandas.core.base import DataError

from floris.utilities import wrap_360

from floris_scada_analysis import floris_tools as ftools
from floris_scada_analysis import optimization as opt
from floris_scada_analysis import scada_analysis as sca
from floris_scada_analysis import time_operations as tops


def printnow(text):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(now_time + ': ' + text)
    return True


class bias_estimation():
    def __init__(self, df, df_fi, fi, test_turbines_subset,
                 sliding_window_lb_ub, df_upstream=None,
                 eo_ws_step=5.0, eo_wd_step=2.0, verbose=False):
        printnow('Initializing the bias_correction() class from floris_scada_analysis.')

        self.verbose = verbose

        # Check format of dataframes based on first 100 rows
        if df.shape[0] != df_fi.shape[0]:
            raise DataError('Please ensure df_fi and df have the same number of entries.')
        if not (list(df.iloc[0:100]['time'])==list(df_fi.iloc[0:100]['time'])):
            raise DataError("The columns 'time' should be identical between df and df_fi.")
        if not (list(df.iloc[0:100]['wd'])==list(df_fi.iloc[0:100]['wd'])):
            raise DataError("The columns 'wd' should be identical between df and df_fi.")
        if not (list(df.iloc[0:100]['ws'])==list(df_fi.iloc[0:100]['ws'])):
            raise DataError("The columns 'ws' should be identical between df and df_fi.")

        df = df.reset_index(drop=('time' in df.columns))
        df_fi = df_fi.reset_index(drop=('time' in df_fi.columns))

        # Import
        self.fi = fi  # Floris object
        self.df = df  # Actual measurements
        self.df_fi = df_fi  # Floris predictions
        self.df_subset = None # Subset of interest (actual)
        self.df_fi_subset = None # Subset of interest (floris)

        self.turbine_list = range(len(fi.layout_x))
        self.test_turbines_subset = test_turbines_subset
        # self.ref_turbine_maxrange = ref_turbine_maxrange
        self.eo_wd_step = eo_wd_step
        self.eo_ws_step = eo_ws_step

        self._reset_results()
        self.opt_search_range = (-180., 180.)
        self.opt_search_dx = 0.50

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

    def estimate_wd_bias(self, N_btstrp=1):
        printnow('Estimating the wind direction bias')
        self._get_energy_ratios_allbins(N_btstrp=N_btstrp)

        # printnow('  Combining %d turbine energy ratios into single curve'
        #          % len(self.test_turbines_subset))
        # x_array = []
        # y_scada = []
        # y_floris = []
        # for ii in range(len(self.test_turbines_subset)):
        #     x_array.extend(ii*400. + np.array(self.energy_ratio_wd_total[ii]))  # Space curves apart by > 360.
        #     y_scada.extend(self.energy_ratio_scada_total[ii])
        #     y_floris.extend(self.energy_ratio_floris_total[ii])

        # printnow('  Matching SCADA and FLORIS energy ratio curves...')
        # wd_bias, success = opt.find_bias_x(x_array, y_scada, x_array, y_floris,
        #                                    search_range = self.opt_search_range,
        #                                    search_dx = self.opt_search_dx)

        # Unpack variables
        wd_arrays = [self.energy_ratios_scada[i]['wd_bin'] for i
                     in range(len(self.energy_ratios_scada))]
        er_result_scada = [self.energy_ratios_scada[i]['baseline'] for i
                           in range(len(self.energy_ratios_scada))]
        er_result_floris = [self.energy_ratios_floris[i]['baseline'] for i
                            in range(len(self.energy_ratios_floris))]

        wd_bias, success = opt.find_wd_bias_by_energy_ratios(
            er_wd_list=wd_arrays,
            er_scada_list=er_result_scada,
            er_floris_list=er_result_floris,
            search_range=self.opt_search_range,
            search_dx=self.opt_search_dx
            )

        self.opt_wd_bias = wd_bias
        self.opt_success = success

        return wd_bias, success

    def _update_df_subset(self):
        printnow('Updated dataframes according to sliding window lock.')
        self.df_subset = self.df.loc[self._sw_lower_idx:self._sw_upper_idx].copy()
        self.df_fi_subset = self.df_fi.loc[self._sw_lower_idx:self._sw_upper_idx].copy()

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

        # # Now mapping upstream turbine ids to test turbines
        # x_turbs = fi.layout_x
        # y_turbs = fi.layout_y
        # test_turbines = self.test_turbines_subset
        # max_radius = self.ref_turbine_maxrange
        # upstream_turbs_ids_per_test_turb = [[] for _ in range(len(test_turbines))]
        # for ii in range(len(test_turbines)):
        #     ti = test_turbines[ii]
        #     printnow('    Mapping upstream turbines per wind direction to each individual test turbine, ti = %d' % (ti))
        #     turbs_in_radius = ftools.get_turbs_in_radius(x_turbs, y_turbs,
        #                                                  ti, max_radius,
        #                                                  include_itself=include_itself)

        #     for wd_bin_ii in range(len(self.upstream_turbs_wds)):
        #         upstream_turbs_local = [tii for tii in self.upstream_turbs_ids[wd_bin_ii] if (tii in turbs_in_radius)]
        #         upstream_turbs_ids_per_test_turb[ii].append(upstream_turbs_local)

        # # Save to self
        # self.upstream_turbs_ids_per_test_turb = upstream_turbs_ids_per_test_turb

    def _get_energy_ratios_allbins(self, N_btstrp=1):
        wd_step = self.eo_wd_step
        ws_step = self.eo_ws_step

        fsc = sca.scada_analysis(verbose=self.verbose)
        fsc.add_df(self.df_subset, 'Measurement data')
        fsc.add_df(self.df_fi_subset, 'FLORIS predictions')

        test_turbines = self.test_turbines_subset
        self.energy_ratios_scada = [[] for _ in test_turbines]
        self.energy_ratios_floris = [[] for _ in test_turbines]

        for ii in range(len(test_turbines)):
            ti = test_turbines[ii]
            printnow('  Determining energy ratio for test turbine = %03d' % (ti))
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
            if not np.isnan(self.opt_wd_bias):
                x_corrected = wrap_360(wd_arrays[ii] - self.opt_wd_bias)
                y_corrected = np.array(er_result_scada[ii])[np.argsort(x_corrected)]
                y_corrected_l = np.array(er_result_scada_l[ii])[np.argsort(x_corrected)]
                y_corrected_u = np.array(er_result_scada_u[ii])[np.argsort(x_corrected)]
                x_corrected = np.sort(x_corrected)
                ax.plot(x_corrected, y_corrected, color='blue', label='SCADA data (corrected)')
                ax.fill_between(x_corrected, y_corrected_l, y_corrected_u, alpha=0.15)
            ax.plot(wd_arrays[ii], er_result_floris[ii], ls = '--', color='orange', label='FLORIS')
            ax.fill_between(wd_arrays[ii], er_result_floris_l[ii], er_result_floris_u[ii], alpha=0.15)
            plt.title('Turbine %d. Current time: %s' % (ti, str(self.sw_current_time)))
            plt.ylabel('Energy ratio (-)')
            plt.xlabel('Wind direction (deg)')
            plt.grid(b=True, which='major', axis='both', color='gray')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', axis='both', color='lightgray')
            plt.legend()

            if save_path is not None:
                plt.savefig(save_path + '_%03d.%s' % (ti, format))
