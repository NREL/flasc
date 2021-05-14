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
import os as os
from scipy import optimize as opt
from scipy import stats as spst

from floris.utilities import wrap_360

from floris_scada_analysis.logging import printnow as print
from floris_scada_analysis import floris_tools as ftools
from floris_scada_analysis import scada_analysis as sca


class bias_estimation():
    def __init__(self,
                 df,
                 df_fi_approx,
                 test_turbines_subset,
                 df_ws_mapping_func,
                 df_pow_ref_mapping_func,
                 verbose=False):
        print('Initializing a bias_estimation() object...')

        # Import inputs
        self.df = df.reset_index(drop=('time' in df.columns))
        self.df_fi_approx = df_fi_approx
        self.verbose = verbose
        self.df_ws_mapping_func = df_ws_mapping_func
        self.df_pow_ref_mapping_func = df_pow_ref_mapping_func
        self.test_turbines_subset = test_turbines_subset

        self.opt_wd_bias = np.nan

    def _load_fsc_for_wd_bias(self, wd_bias):
        print('Initializing fsc class for wd_bias = %.2f deg.'
              % wd_bias)
        df_cor = self.df.copy()
        df_cor['wd'] = wrap_360(df_cor['wd'] - wd_bias)

        # Set columns 'ws' and 'pow_ref' for df_subset_cor
        df_cor = self.df_ws_mapping_func(df_cor)
        df_cor = self.df_pow_ref_mapping_func(df_cor)
        df_cor = df_cor.dropna(subset=['wd', 'ws', 'pow_ref'])

        # Get FLORIS predictions
        df_fi = df_cor[['time', 'wd', 'ws']].copy()
        df_fi = ftools.interpolate_floris_from_df_approx(
            df=df_fi, df_approx=self.df_fi_approx)
        df_fi = self.df_pow_ref_mapping_func(df_fi)

        # Initialize SCADA analysis class and add dataframes
        fsc = sca.scada_analysis(verbose=self.verbose)
        fsc.add_df(df_cor, 'Measurement data')
        fsc.add_df(df_fi, 'FLORIS predictions')

        # Save to self
        self.fsc = fsc
        self.fsc_wd_bias = wd_bias

        return fsc

    def _get_energy_ratios_allbins(self,
                                   wd_bin_size=2.0,
                                   ws_bin_size=3.0,
                                   N_btstrp=1,
                                   plot_iter_path=None):

        test_turbines = self.test_turbines_subset
        energy_ratios_scada = [[] for _ in test_turbines]
        energy_ratios_floris = [[] for _ in test_turbines]

        fsc = self.fsc
        for ii, ti in enumerate(test_turbines):
            print('  Determining energy ratio for test turbine = %03d.'
                  % (ti) + ' WD bias: %.3f deg.' % self.fsc_wd_bias)
            fsc.get_energy_ratios(test_turbines=[ti],
                                  wd_step=wd_bin_size,
                                  ws_step=ws_bin_size,
                                  N=N_btstrp)
            energy_ratios_scada[ii] = fsc.df_list[0]['er_results']
            energy_ratios_floris[ii] = fsc.df_list[1]['er_results']

        # Save to self
        self.energy_ratios_scada = energy_ratios_scada
        self.energy_ratios_floris = energy_ratios_floris

        # Debugging: show all possible options
        if plot_iter_path is not None:
            fp = os.path.join(plot_iter_path, 'bias%+.3f' % (self.fsc_wd_bias),
                              'energyratio')
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            self.plot_energy_ratios(save_path=fp, format='png')
            plt.close('all')

        return None

    def estimate_wd_bias(self,
                         time_mask=None,
                         ws_mask=(6., 10.),
                         wd_mask=None,
                         ti_mask=None,
                         opt_search_range=(-180., 180.),
                         opt_search_brute_dx=5.0,
                         opt_finish=opt.fmin,
                         energy_ratio_wd_binsize=2.0,
                         energy_ratio_ws_binsize=3.0,
                         energy_ratio_N_btstrp=1,
                         plot_iter_path=None):

        print('Estimating the wind direction bias')
        self.time_mask = time_mask

        def cost_fun(wd_bias):
            self._load_fsc_for_wd_bias(wd_bias=wd_bias)
            self.fsc.set_masks(time_range=time_mask,
                               ws_range=ws_mask,
                               wd_range=wd_mask,
                               ti_range=ti_mask)

            self._get_energy_ratios_allbins(
                wd_bin_size=energy_ratio_wd_binsize,
                ws_bin_size=energy_ratio_ws_binsize,
                plot_iter_path=plot_iter_path)

            # Unpack variables
            energy_ratios_scada = self.energy_ratios_scada
            energy_ratios_floris = self.energy_ratios_floris

            # Calculate cost
            cost_array = np.full(len(energy_ratios_scada), np.nan)
            for ii in range(len(energy_ratios_scada)):
                y_scada = np.array(energy_ratios_scada[ii]['baseline'])
                y_floris = np.array(energy_ratios_floris[ii]['baseline'])
                ids = ~np.isnan(y_scada) & ~np.isnan(y_floris)
                if np.sum(ids) > 5:  # At least 6 valid data entries
                    r, _ = spst.pearsonr(y_scada[ids], y_floris[ids])
                else:
                    r = np.nan
                cost_array[ii] = -1. * r

            cost = np.nanmean(cost_array)
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
            Ns=int(np.ceil(dran/opt_search_brute_dx) + 1),
            full_output=True,
            disp=True,
            finish=opt_finish)

        wd_bias = x_opt
        self.opt_wd_bias = wd_bias
        self.opt_cost = J_opt
        self.opt_wd_grid = x
        self.opt_wd_cost = J

        # End with optimal results and bootstrapping
        self._load_fsc_for_wd_bias(wd_bias=x_opt)
        self.fsc.set_masks(time_range=time_mask,
                           ws_range=ws_mask,
                           wd_range=wd_mask,
                           ti_range=ti_mask)
        self._get_energy_ratios_allbins(
            wd_bin_size=energy_ratio_wd_binsize,
            ws_bin_size=energy_ratio_ws_binsize,
            plot_iter_path=plot_iter_path)

        return x_opt, J_opt

    def plot_energy_ratios(self, save_path=None, format='png'):
        # Unpack variables
        energy_ratios_scada = self.energy_ratios_scada
        energy_ratios_floris = self.energy_ratios_floris

        x = [v['wd_bin'] for v in energy_ratios_scada]
        y_scada = [v['baseline'] for v in energy_ratios_scada]
        y_scada_l = [v['baseline_l'] for v in energy_ratios_scada]
        y_scada_u = [v['baseline_u'] for v in energy_ratios_scada]
        y_bins_N = [v['N_bin'] for v in energy_ratios_scada]
        y_floris = [v['baseline'] for v in energy_ratios_floris]
        y_floris_l = [v['baseline_l'] for v in energy_ratios_floris]
        y_floris_u = [v['baseline_u'] for v in energy_ratios_floris]

        print('Plotting results for last evaluated case')
        print('  Wind direction bias: %.1f deg' % (self.opt_wd_bias))
        for ii in range(len(self.test_turbines_subset)):
            ti = self.test_turbines_subset[ii]
            fig, ax = plt.subplots(figsize=(10, 6), nrows=2, sharex=True)
            ax[0].plot(x[ii], y_scada[ii], color='k', label='SCADA data')
            ax[0].fill_between(
                x[ii], y_scada_l[ii], y_scada_u[ii], alpha=0.15
                )
            ax[0].plot(
                x[ii], y_floris[ii], ls='--', color='orange', label='FLORIS'
                )
            ax[0].fill_between(
                x[ii], y_floris_l[ii], y_floris_u[ii], alpha=0.15
                )
            ax[0].set_title(
                'Turbine %d. Time range: %s to %s.'
                % (ti, str(self.time_mask[0]), str(self.time_mask[1]))
                )
            ax[0].set_ylabel('Energy ratio (-)')
            ax[0].grid(b=True, which='major', axis='both', color='gray')
            ax[0].grid(b=True, which='minor', axis='both', color='lightgray')
            ax[0].minorticks_on()
            ax[0].legend()

            ax[1].bar(x[ii], y_bins_N[ii], width=.7*np.diff(x[ii])[0],
                      label='Number of data points', color='black')
            ax[1].grid(b=True, which='major', axis='both', color='gray')
            ax[1].grid(b=True, which='minor', axis='both', color='lightgray')
            ax[1].set_xlabel('Wind direction (deg)')
            ax[1].set_ylabel('Number of data points (-)')

            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path + '_%03d.%s' % (ti, format))
