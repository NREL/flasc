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

from floris_scada_analysis.utilities import printnow as print
from floris_scada_analysis import floris_tools as ftools
from floris_scada_analysis.energy_ratio import energy_ratio_suite


class bias_estimation():
    """This class can be used to estimate the bias (offset) in a wind
    direction measurement by comparing the energy ratios in the SCADA
    data with the predicted energy ratios from FLORIS under various
    bias correction values. Essentially, this class solves the following
    optimization problem: the argument in this optimization problem
    is the offset on the wind direction measurement and the cost is the
    Pearson correlation coefficient between two lines, namely
    1) the energy ratios of the SCADA data, and 2) the predicted energy
    ratios for the same data by FLORIS under the offset-corrected wind
    direction.
    """
    def __init__(
        self,
        df,
        df_fi_approx,
        test_turbines_subset,
        df_ws_mapping_func,
        df_pow_ref_mapping_func
    ):
        """Initialize the bias estimation class.

        Args:
            df ([pd.DataFrame]): Dataframe with the SCADA data measurements
                formatted in the generic format. The dataframe should contain
                at the minimum the following columns:
                * Reference wind direction for the test turbine, 'wd'
                * Reference wind speed for the test turbine, 'ws'
                * Power production of every turbine: pow_000, pow_001, ...
                * Reference power production used to normalize the energy
                    ratio: 'pow_ref'
            df_fi_approx ([pd.DataFrame]): Dataframe containing a large set
                of precomputed solutions of the FLORIS model for a range of
                wind directions, wind speeds (and optionally turbulence
                intensities). This table can be generated using the following:
                    from floris_scada_analysis import floris_tools as ftls
                    df_approx = ftls.calc_floris_approx_table(...)
            test_turbines_subset ([iteratible]): List of test turbines for
                which each the energy ratios are calculated and the Pearson
                correlation coefficients are calculated. Note that this
                variable is slightly different from 'test_turbines' in the
                energy ratio classes. Namely, in this class, the energy ratio
                is calculated for each entry in test_turbines_subset
                separately, while in the other classes one energy ratio is
                calculated based on the mean power production of all turbines
                in test_turbines.
            df_ws_mapping_func ([function]): This is a function that
                returns the reference wind speed based on an array of wind
                directions as input.
            df_pow_ref_mapping_func ([type]): This is a function that
                returns the reference power production based on an array of
                wind directions as input.
        """
        print('Initializing a bias_estimation() object...')

        # Import inputs
        self.df = df.reset_index(drop=('time' in df.columns))
        self.df_fi_approx = df_fi_approx
        self.df_ws_mapping_func = df_ws_mapping_func
        self.df_pow_ref_mapping_func = df_pow_ref_mapping_func
        self.test_turbines_subset = test_turbines_subset

    # Private methods

    def _load_ersuite_for_wd_bias(self, wd_bias):
        """This function initializes an instance of the energy_ratio_suite
        class where the dataframe is shifted by wd_bias. This facilitates
        the calculation of the energy ratios under this hypothesized wind
        direction bias. Additionally, the FLORIS predictions for the shifted
        dataset are calculated and the energy ratios for the FLORIS
        predictions are also calculated.

        Args:
            wd_bias ([float]): Hypothesized wind direction bias in degrees.

        Returns:
            fsc ([energy_ratio_suite object]): The energy ratio suite
                object in which the inserted dataframe has a shifted
                wind direction measurement, offset by 'wd_bias' compared
                to the nominal dataset.
        """
        print('  Calculating cost metric for an offset of %.2f deg.'
              % wd_bias)
        df_cor = self.df.copy()
        df_cor['wd'] = wrap_360(df_cor['wd'] - wd_bias)

        # Set columns 'ws' and 'pow_ref' for df_subset_cor
        df_cor = self.df_ws_mapping_func(df_cor)
        df_cor = self.df_pow_ref_mapping_func(df_cor)
        df_cor = df_cor.dropna(subset=['wd', 'ws', 'pow_ref'])

        # Get FLORIS predictions
        print('    Interpolating FLORIS predictions for dataframe.')
        df_fi = df_cor[['time', 'wd', 'ws']].copy()
        df_fi = ftools.interpolate_floris_from_df_approx(
            df=df_fi, df_approx=self.df_fi_approx, verbose=False)
        df_fi = self.df_pow_ref_mapping_func(df_fi)

        # Initialize SCADA analysis class and add dataframes
        fsc = energy_ratio_suite.energy_ratio_suite(verbose=False)
        fsc.add_df(df_cor, 'Measurement data')
        fsc.add_df(df_fi, 'FLORIS predictions')

        # Save to self
        self.fsc = fsc
        self.fsc_wd_bias = wd_bias

        return fsc

    def _get_energy_ratios_allbins(
        self,
        wd_step=3.0,
        ws_step=1.0,
        wd_bin_width=3.0,
        N_btstrp=1,
        plot_iter_path=None
    ):
        """Calculate the energy ratios for the energy_ratio_suite object
        contained in 'self.fsc'.

        Args:
            wd_step (float, optional): Wind direction discretization step
                size. This defines for what wind directions the energy ratio
                is to be calculated. Note that this does not necessarily
                also mean each bin has a width of this value. Namely, the
                bin width can be specified separately. Defaults to 2.0.
            ws_step (float, optional): Wind speed discretization step size.
                This defines the resolution and widths of the wind speed
                bins. Defaults to 1.0.
            wd_bin_width ([type], optional): The wind direction bin width.
                This value should be equal or larger than wd_step. When no
                value is specified, will default to wd_bin_width = wd_step.
                In the literature, it is not uncommon to specify a bin width
                larger than the step size to cover for variability in the
                wind direction measurements. By setting a large value for
                wd_bin_width, one gets a better idea of the larger-scale
                wake losses in the wind farm. Defaults to None.
            N_btstrp (int, optional): Number of bootstrap evaluations for
                uncertainty quantification (UQ). If N_btstrp=1, will not
                perform any uncertainty quantification. Defaults to 1.
            plot_iter_path ([type], optional): Path to save figures of the
                energy ratios of each iteration to. If not specified, will
                not plot or save any figures of iterations. Defaults to
                None.
        """
        test_turbines = self.test_turbines_subset
        energy_ratios_scada = [[] for _ in test_turbines]
        energy_ratios_floris = [[] for _ in test_turbines]

        fsc = self.fsc
        for ii, ti in enumerate(test_turbines):
            print('    Determining energy ratios for test turbine = %03d.'
                  % (ti) + ' WD bias: %.3f deg.' % self.fsc_wd_bias)
            fsc.get_energy_ratios(
                test_turbines=[ti],
                wd_step=wd_step,
                ws_step=ws_step,
                wd_bin_width=wd_bin_width,
                N=N_btstrp
            )
            energy_ratios_scada[ii] = fsc.df_list[0]['er_results']
            energy_ratios_floris[ii] = fsc.df_list[1]['er_results']

        # Save to self
        self.energy_ratios_scada = energy_ratios_scada
        self.energy_ratios_floris = energy_ratios_floris

        # Debugging: show all possible options
        if plot_iter_path is not None:
            print('    Plotting & saving energy ratios for this iteration')
            fp = os.path.join(
                plot_iter_path,
                'bias%+.3f' % (self.fsc_wd_bias),
                'energyratio'
                )
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            self.plot_energy_ratios(save_path=fp, format='png')
            plt.close('all')

        return None

    # Public methods

    def calculate_baseline(
        self,
        time_mask=None,
        ws_mask=(6.0, 10.0),
        wd_mask=None,
        ti_mask=None,
        er_wd_step=3.0,
        er_ws_step=5.0,
        er_wd_bin_width=None,
        er_N_btstrp=1,
    ):
        self._load_ersuite_for_wd_bias(wd_bias=0.0)
        self.fsc.set_masks(
            time_range=time_mask,
            ws_range=ws_mask,
            wd_range=wd_mask,
            ti_range=ti_mask
        )
        self._get_energy_ratios_allbins(
            wd_step=er_wd_step,
            ws_step=er_ws_step, 
            wd_bin_width=er_wd_bin_width,
            N_btstrp=er_N_btstrp
        )

    def estimate_wd_bias(
        self,
        time_mask=None,
        ws_mask=(6.0, 10.0),
        wd_mask=None,
        ti_mask=None,
        opt_search_range=(-180.0, 180.0),
        opt_search_brute_dx=5,
        er_wd_step=3.0,
        er_ws_step=5.0,
        er_wd_bin_width=None,
        er_N_btstrp=1,
        plot_iter_path=None
    ):
        """Estimate the wind direction bias by comparing the SCADA data
        under various wind direction corrections to its FLORIS predictions.

        Args:
            time_mask ([iterable], optional): Wind speed mask. Should be an
                iterable of length 2, e.g., [pd.to_datetime("2019-01-01"),
                pd.to_datetime("2019-04-01")], defining the lower and upper
                bound, respectively. If not specified, will not mask the data
                based on this variable. Defaults to None.
            ws_mask ([iterable], optional): Wind speed mask. Should be an
                iterable of length 2, e.g., [6.0, 10.0], defining the lower
                and upper bound, respectively. If not specified, will not
                mask the data based on this variable. Defaults to (6, 10).
            wd_mask ([iterable], optional): Wind direction mask. Should
                be an iterable of length 2, e.g., [0.0, 180.0], defining
                the lower and upper bound, respectively. If not specified,
                will not mask the data based on this variable. Defaults to
                None.
            ti_mask ([iterable], optional): Turbulence intensity mask.
                Should be an iterable of length 2, e.g., [0.04, 0.08],
                defining the lower and upper bound, respectively. If not
                specified, will not mask the data based on this variable.
                Defaults to None.
            opt_search_range (tuple, optional): Search range for the wind
                direction offsets to consider. Defaults to (-180., 180.).
            opt_search_brute_dx (float, optional): Number of points to
                discretize the search space over. Defaults to 5.
            er_wd_step (float, optional): Wind direction discretization step
                size. This defines for what wind directions the energy ratio
                is to be calculated. Note that this does not necessarily
                also mean each bin has a width of this value. Namely, the
                bin width can be specified separately. Defaults to 3.0.
            er_ws_step (float, optional): Wind speed discretization step size.
                This defines the resolution and widths of the wind speed
                bins. Defaults to 5.0.
            er_wd_bin_width ([type], optional): The wind direction bin width.
                This value should be equal or larger than wd_step. When no
                value is specified, will default to wd_bin_width = wd_step.
                In the literature, it is not uncommon to specify a bin width
                larger than the step size to cover for variability in the
                wind direction measurements. By setting a large value for
                wd_bin_width, one gets a better idea of the larger-scale
                wake losses in the wind farm. Defaults to None.
            er_N_btstrp (int, optional): Number of bootstrap evaluations for
                uncertainty quantification (UQ). If N_btstrp=1, will not
                perform any uncertainty quantification. Defaults to 1.
            plot_iter_path ([type], optional): Path to save figures of the
                energy ratios of each iteration to. If not specified, will
                not plot or save any figures of iterations. Defaults to
                None.

        Returns:
            x_opt ([float]): Optimal wind direction offset.
            J_opt ([float]): Cost function under optimal offset.
        """
        print('Estimating the wind direction bias')

        def cost_fun(wd_bias):
            self._load_ersuite_for_wd_bias(wd_bias=wd_bias)
            self.fsc.set_masks(time_range=time_mask,
                               ws_range=ws_mask,
                               wd_range=wd_mask,
                               ti_range=ti_mask)

            self._get_energy_ratios_allbins(
                wd_step=er_wd_step,
                ws_step=er_ws_step,
                wd_bin_width=er_wd_bin_width,
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

        opt_finish = (
            lambda func, x0, args=(): opt.fmin(func, x0, args,
                                               maxfun=10,
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
        print('  Evaluating optimal solution with bootstrapping')
        self._load_ersuite_for_wd_bias(wd_bias=x_opt)
        self.fsc.set_masks(time_range=time_mask,
                           ws_range=ws_mask,
                           wd_range=wd_mask,
                           ti_range=ti_mask)

        self._get_energy_ratios_allbins(
            wd_step=er_wd_step,
            ws_step=er_ws_step,
            wd_bin_width=er_wd_bin_width,
            N_btstrp=er_N_btstrp,
            plot_iter_path=None)

        return x_opt, J_opt

    def plot_energy_ratios(self, save_path=None, format='png', dpi=200):
        """Plot the energy ratios for the currently evaluated wind
        direction offset term.

        Args:
            save_path ([str], optional): Path to save the figure to. If not
                specified, will not save the figure. Defaults to None.
            format (str, optional): Figure format. Defaults to 'png'.
            dpi (int, optional): Figure DPI. Defaults to 200.
        """
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
        time_range = (list(self.fsc.df_list[0]['df']['time'])[0],
                      list(self.fsc.df_list[0]['df']['time'])[-1])

        fig_list = []
        ax_list = []
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
                % (ti, str(time_range[0]), str(time_range[1]))
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
                plt.savefig(save_path + '_%03d.%s' % (ti, format), dpi=dpi)

            fig_list.append(fig)
            ax_list.append(ax)

        return fig_list, ax_list
