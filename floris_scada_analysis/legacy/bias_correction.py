import copy
import datetime
import numpy as np
from scipy import stats as spst
import pandas as pd
import floris_scada as fsc


def printnow(text):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(now_time + ': ' + text)
    return True


def find_bias_x(x_1, y_1, x_2, y_2, search_range, search_dx):
    x_1 = np.array(x_1)
    y_2 = np.array(y_2)

    def errfunc(dx, x_1, x_2, y_1, y_2):
        y_1_cor = np.interp(x_2, x_1 - dx, y_1)
        y_1_isnan = np.isnan(y_1_cor)
        y_2_isnan = np.isnan(y_2)        

        # Clean up data
        clean_data = [~a and ~b for a, b in zip(y_1_isnan, y_2_isnan)]
        y_1_cor = y_1_cor[clean_data]
        y_2 = y_2[clean_data]

        if all(np.isnan(y_1_cor)) and all(np.isnan(y_2)):
            cost = np.nan
        else:
            cost = -1.0 * spst.pearsonr(y_1_cor, y_2)[0]
        return cost

    printnow('Using brute force to estimate the wind direction bias...')
    cost_min = 1.0e15
    success = False
    p1 = [0.0]
    for dx_opt in np.arange(search_range[0], search_range[1], search_dx):
        cost_eval = errfunc(dx_opt, x_1, x_2, y_1, y_2)
        if cost_eval <= cost_min:
            p1 = [dx_opt]
            cost_min = cost_eval
            success = True
    dx_opt = p1[0]

    return dx_opt, success


class bias_estimation():
    def __init__(self, fs, test_turbines_subset, ref_turbine_maxrange,
                 ws_range, sliding_window_lb_ub, eo_ws_step=5.0, eo_wd_step=2.0):
        printnow('Initializing the bias_correction() class from floris_scada.')
        self.fs = fs
        self.fi = fs.fi_array[0]
        self.df_full = self.fs.df

        self.turbine_list = range(len(self.fi.get_turbine_layout()[0]))
        self.test_turbines_subset = test_turbines_subset
        self.ref_turbine_maxrange = ref_turbine_maxrange
        self.ws_range = ws_range
        self.eo_wd_step = eo_wd_step
        self.eo_ws_step = eo_ws_step

        self._reset_results()
        self.opt_search_range = (-50., 50.)
        self.opt_search_dx = 0.10

        self._get_ref_turbs_floris(wd_step=self.eo_wd_step)

        printnow('  Initializing sliding window')
        time_array = list(pd.to_datetime(fs.df.time))
        self.sw_time_array = time_array
        self.sw_current_time = time_array[0]
        self.sw_lb_ub = sliding_window_lb_ub
        dt_array = [(time_array[i+1] - time_array[i])/datetime.timedelta(seconds=1) for i in range(np.min([5000, len(time_array)-1]))]
        self.sw_time_array_dt = datetime.timedelta(seconds=np.nanmin(dt_array))
        printnow('  Estimated the data to be sampled at %.1f seconds.' % (self.sw_time_array_dt.seconds))

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

    def estimate_wd_bias(self):
        printnow('Estimating the wind direction bias')
        self._get_energy_ratios_allbins()

        printnow('  Combining %d turbine energy ratios into single curve' % len(self.test_turbines_subset))
        x_array = []
        y_scada = []
        y_floris = []
        for ii in range(len(self.test_turbines_subset)):
            x_array.extend(ii*400. + np.array(self.energy_ratio_wd_total[ii]))  # Space curves apart by > 360.
            y_scada.extend(self.energy_ratio_scada_total[ii])
            y_floris.extend(self.energy_ratio_floris_total[ii])

        printnow('  Matching SCADA and FLORIS energy ratio curves...')
        wd_bias, success = find_bias_x(x_array, y_scada, x_array, y_floris,
                                       search_range = self.opt_search_range,
                                       search_dx = self.opt_search_dx)

        self.opt_wd_bias = wd_bias
        self.opt_success = success

        return wd_bias, success

    def plot_energy_ratios(self):
        import matplotlib.pyplot as plt

        wd_arrays = self.energy_ratio_wd_total
        er_result_scada = self.energy_ratio_scada_total
        er_result_floris = self.energy_ratio_floris_total

        printnow('Plotting results for last evaluated case')
        printnow('  Wind direction bias: %.1f deg' % (self.opt_wd_bias))
        for ii in range(len(self.test_turbines_subset)):
            ti = self.test_turbines_subset[ii]
            fig, ax = plt.subplots()
            ax.plot(wd_arrays[ii], er_result_scada[ii], color='k', label='SCADA data')
            if not np.isnan(self.opt_wd_bias):
                ax.plot(wd_arrays[ii] - self.opt_wd_bias, er_result_scada[ii], color='blue', label='SCADA data (corrected)')
            ax.plot(wd_arrays[ii], er_result_floris[ii], ls = '--', color='orange', label='FLORIS')
            plt.title('Turbine %d' % ti)
        plt.legend()
        plt.show()

    def _update_df_subset(self):
        printnow('Updated dataframe according to sliding window lock.')
        self.fs.df = self.df_full.loc[self._sw_lower_idx:self._sw_upper_idx]

    def _reset_results(self):
        # Set-up outputs
        self.energy_ratio_wd_total = []
        self.energy_ratio_scada_total = []
        self.energy_ratio_floris_total = []

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
            self._sw_lower_idx = np.max([self._sw_lower_idx, 0])
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
    def _get_ref_turbs_floris(self, wd_step):
        printnow('  Determining upstream turbines per wind direction using FLORIS for wd_step = %.1f deg.' % (wd_step))
        fi = self.fi
        upstream_turbs_ids = []  # turbine numbers that are freestream
        upstream_turbs_wds = []  # lower bound of bin
        for wd in np.arange(0., 360., wd_step):
            fi.reinitialize_flow_field(wind_direction=wd, wind_speed=8.0)
            fi.calculate_wake()
            power_out = np.array(fi.get_turbine_power())
            power_wake_loss = np.max(power_out) - power_out
            turbs_freestream = list(np.where(power_wake_loss < 0.01)[0])
            if len(upstream_turbs_wds) == 0:
                upstream_turbs_ids.append(turbs_freestream)
                upstream_turbs_wds.append(wd)
            elif not(turbs_freestream == upstream_turbs_ids[-1]):
                upstream_turbs_ids.append(turbs_freestream)
                upstream_turbs_wds.append(wd)

        # Connect at 360 degrees
        if upstream_turbs_ids[0] == upstream_turbs_ids[-1]:
            upstream_turbs_wds.pop(0)
            upstream_turbs_ids.pop(0)

        # Go from list to bins for upstream_turbs_wds
        upstream_turbs_wds = [[upstream_turbs_wds[i], upstream_turbs_wds[i+1]] for i in range(len(upstream_turbs_wds)-1)]
        upstream_turbs_wds.append([upstream_turbs_wds[-1][-1], 360. + upstream_turbs_wds[0][0]])

        self.upstream_turbs_wds = upstream_turbs_wds
        self.upstream_turbs_ids = upstream_turbs_ids

        # Now mapping upstream turbine ids to test turbines
        x_turbs = fi.layout_x
        y_turbs = fi.layout_y
        test_turbines = self.test_turbines_subset
        max_radius = self.ref_turbine_maxrange
        upstream_turbs_ids_per_test_turb = [[] for _ in range(len(test_turbines))]
        for ii in range(len(test_turbines)):
            ti = test_turbines[ii]
            printnow('    Mapping upstream turbines per wind direction to each individual test turbine, ti = %d' % (ti))
            dr_turb = np.sqrt((x_turbs - x_turbs[ti])**2.0 + (y_turbs - y_turbs[ti])**2.0)
            for wd_bin_ii in range(len(upstream_turbs_wds)):
                turbs_within_radius = np.where(dr_turb <= max_radius)[0]
                upstream_turbs_local = [tii for tii in upstream_turbs_ids[wd_bin_ii] if (tii in turbs_within_radius)]
                upstream_turbs_ids_per_test_turb[ii].append(upstream_turbs_local)

        # Save to self
        self.upstream_turbs_ids_per_test_turb = upstream_turbs_ids_per_test_turb

    def _get_energy_ratios_singlebin(self, test_turbines, ref_turbines, wd_range):
        # printnow('Determining the energy ratio for wd_range (%.1f, %.1f) and ws_range (%.1f, %.1f).'
        #          % (wd_range[0], wd_range[1], self.ws_range[0], self.ws_range[1]))

        # Standardized settings
        control_turbine = []
        dep_turbines = []
        full_filter_all = True

        # Initialize eo object
        eo = fsc.energy_tools.Energy_Analysis(self.fs, test_turbines, ref_turbines,
                                            control_turbines=control_turbine,
                                            dep_turbines=dep_turbines,
                                            wd_step=self.eo_wd_step,
                                            ws_step=self.eo_ws_step,
                                            full_filter_all=full_filter_all,
                                            do_prints=False)

        if eo.df.shape[0] <= 0:
            result = []
            result_floris = []
        else:
            eo.df['control_mode'] = 'baseline'
            eo.df['category'] = 'baseline'

            # Make an Energy Analysis class for floris
            eo_floris = copy.deepcopy(eo)
            eo_floris.df['ref_power'] = eo_floris.df[['floris_original_%03d' % t for t in ref_turbines]].mean(axis=1)
            eo_floris.df['test_power'] = eo_floris.df[['floris_original_%03d' % t for t in test_turbines]].mean(axis=1)

            # Set the energy frames
            eo.get_energy_ratio_frame(self.ws_range, wd_range, ['baseline'])
            eo_floris.get_energy_ratio_frame(self.ws_range, wd_range, ['baseline'])

            # Get the analysis results
            result = eo.energy_frame.get_1_cat_energy_ratio_array_with_range(N=1)
            result_floris = eo_floris.energy_frame.get_1_cat_energy_ratio_array_with_range(N=1)

        return result, result_floris

    def _get_energy_ratios_allbins(self):
        wd_step = self.eo_wd_step

        wd_result_total = np.arange(0., 360., wd_step)
        if np.abs(wd_step-2.) < 1e-5:
            wd_result_total += 1.0 # Quick fix
        elif np.abs(wd_step - 1.0) > 1e-5:
            raise ModuleNotFoundError('Need to figure this out later...')

        test_turbines = self.test_turbines_subset
        # [np.repeat(np.nan, 1) for _ in self.turbine_list]
        er_result_wd = [[] for _ in test_turbines]
        er_result_scada = [[] for _ in test_turbines]
        er_result_floris = [[] for _ in test_turbines]

        for ii in range(len(test_turbines)):
            ti = test_turbines[ii]
            printnow('  Determining energy ratio for test turbine = %03d' % (ti))
            for wd_bin_i in range(len(self.upstream_turbs_wds)):
                wd_range = (self.upstream_turbs_wds[wd_bin_i][0], 
                            self.upstream_turbs_wds[wd_bin_i][1])

                ref_turbines = self.upstream_turbs_ids_per_test_turb[ii][wd_bin_i]
                if (not (ti in ref_turbines) and len(ref_turbines) > 0):  # Ignore non-waked situations
                    result, result_floris = self._get_energy_ratios_singlebin(test_turbines=[ti],
                                                            ref_turbines=ref_turbines,
                                                            wd_range=wd_range)

                    if len(result) > 0:
                        er_result_wd[ii].extend(result.wd_bin)
                        er_result_scada[ii].extend(result.baseline)
                        er_result_floris[ii].extend(result_floris.baseline)

        self.energy_ratio_wd_total = er_result_wd
        self.energy_ratio_scada_total = er_result_scada
        self.energy_ratio_floris_total = er_result_floris
