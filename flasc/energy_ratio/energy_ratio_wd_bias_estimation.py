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

from ..dataframe_operations import dataframe_manipulations as dfm
from .. import floris_tools as ftools
from ..utilities import printnow as print
from ..energy_ratio import energy_ratio_suite


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
                    from flasc import floris_tools as ftls
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
        self.n_turbines = dfm.get_num_turbines(self.df)
        self.df_fi_approx = df_fi_approx
        self.df_ws_mapping_func = df_ws_mapping_func
        self.df_pow_ref_mapping_func = df_pow_ref_mapping_func
        self.test_turbines_subset = test_turbines_subset

    # Private methods

    def _load_ersuites_for_wd_bias(
        self,
        wd_bias,
        test_turbines,
        time_mask=None,
        wd_mask=None,
        ws_mask=None,
        ti_mask=None,
    ):
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
        print('  Constructing energy ratio suites for wd_bias of %.2f deg.'
              % wd_bias)

        fsc_list = []
        fsc_wd_bias_list = []

        # Derive dataframe that covers all test_turbines
        df_cor_all = self.df.copy()
        df_cor_all['wd'] = wrap_360(df_cor_all['wd'] - wd_bias)

        # Set columns 'ws' and 'pow_ref' for df_subset_cor
        df_cor_all = self.df_ws_mapping_func(df_cor_all)
        df_cor_all = self.df_pow_ref_mapping_func(df_cor_all)
        df_cor_all = df_cor_all.dropna(subset=['wd', 'ws', 'pow_ref'])
        df_cor_all = df_cor_all.reset_index(drop=True)

        # Get FLORIS predictions
        print('    Interpolating FLORIS predictions for dataframe.')
        ws_cols = ["ws_{:03d}".format(ti) for ti in range(self.n_turbines)]
        pow_cols = ["pow_{:03d}".format(ti) for ti in range(self.n_turbines)]
        df_fi_all = df_cor_all[['time', 'wd', 'ws', 'ti', *ws_cols, *pow_cols]].copy()

        df_fi_all = ftools.interpolate_floris_from_df_approx(
            df=df_fi_all,
            df_approx=self.df_fi_approx,
            verbose=False,
            mirror_nans=True
        )
        df_fi_all = self.df_pow_ref_mapping_func(df_fi_all)

        for ti in test_turbines:
            valid_entries = (
                (~df_cor_all["pow_{:03d}".format(ti)].isna()) &
                (~df_fi_all["pow_{:03d}".format(ti)].isna())
            )
            df_cor = df_cor_all[valid_entries].copy().reset_index(drop=True)
            df_fi = df_fi_all[valid_entries].copy().reset_index(drop=True)

            # Initialize SCADA analysis class and add dataframes
            fsc = energy_ratio_suite.energy_ratio_suite(verbose=False)
            fsc.add_df(df_cor, 'Measurement data')
            fsc.add_df(df_fi, 'FLORIS predictions')

            fsc.set_masks(
                time_range=time_mask,
                ws_range=ws_mask,
                wd_range=wd_mask,
                ti_range=ti_mask,
            )

            fsc_list.append(fsc)
            fsc_wd_bias_list.append(wd_bias)

        # Save to self
        self.fsc_list = fsc_list
        self.fsc_wd_bias_list = fsc_wd_bias_list
        self.fsc_test_turbine_list = test_turbines

    def _get_energy_ratios_allbins(
        self,
        wd_bias,
        time_mask=None,
        ws_mask=(6.0, 10.0),
        wd_mask=None,
        ti_mask=None,
        wd_step=3.0,
        ws_step=1.0,
        wd_bin_width=3.0,
        N_btstrp=1,
        plot_iter_path=None,
        fast=True,
    ):
        """Calculate the energy ratios for the energy_ratio_suite objects
        contained in 'self.fsc_list'.

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

        print("    Initializing energy ratio suites.")
        self._load_ersuites_for_wd_bias(
            wd_bias=wd_bias,
            test_turbines=test_turbines,
            time_mask=time_mask,
            ws_mask=ws_mask,
            wd_mask=wd_mask,
            ti_mask=ti_mask,
        )

        for ii, ti in enumerate(test_turbines):
            print('    Determining energy ratios for test turbine = %03d.'
                  % (ti) + ' WD bias: %.3f deg.' % self.fsc_wd_bias_list[ii])
            fsc = self.fsc_list[ii]
            if fast == True:
                fsc.get_energy_ratios_fast(
                    test_turbines=[ti],
                    wd_step=wd_step,
                    ws_step=ws_step,
                    wd_bin_width=wd_bin_width,
                    verbose=False,
                )
            else:
                fsc.get_energy_ratios(
                    test_turbines=[ti],
                    wd_step=wd_step,
                    ws_step=ws_step,
                    wd_bin_width=wd_bin_width,
                    N=N_btstrp,
                    balance_bins_between_dfs=False,
                    verbose=False,
                )
            energy_ratios_scada[ii] = fsc.df_list[0]['er_results']
            energy_ratios_floris[ii] = fsc.df_list[1]['er_results']

        # Debugging: plot iteration to path
        if plot_iter_path is not None:
            print('    Plotting energy ratios and saving figures')
            fp = os.path.join(
                plot_iter_path,
                "bias%+.3f" % (self.fsc_wd_bias_list[ii]),
                "energy_ratios_test_turbine")
            self.plot_energy_ratios(save_path=fp, format='png', dpi=200)
            plt.close('all')

        # Save to self
        self.energy_ratios_scada = energy_ratios_scada
        self.energy_ratios_floris = energy_ratios_floris

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
        self._get_energy_ratios_allbins(
            wd_bias=0.0,
            time_mask=time_mask,
            ws_mask=ws_mask,
            wd_mask=wd_mask,
            ti_mask=ti_mask,
            wd_step=er_wd_step,
            ws_step=er_ws_step, 
            wd_bin_width=er_wd_bin_width,
            N_btstrp=er_N_btstrp,
            fast=False,
        )

    def estimate_wd_bias(
        self,
        time_mask=None,
        ws_mask=(6.0, 10.0),
        wd_mask=None,
        ti_mask=None,
        opt_search_range=(-180.0, 180.0),
        opt_search_brute_dx=5,
        opt_workers=4,
        er_wd_step=3.0,
        er_ws_step=5.0,
        er_wd_bin_width=None,
        er_N_btstrp=1,
        plot_iter_path=None,
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
            self._get_energy_ratios_allbins(
                wd_bias=wd_bias,
                time_mask=time_mask,
                ws_mask=ws_mask,
                wd_mask=wd_mask,
                ti_mask=ti_mask,
                wd_step=er_wd_step,
                ws_step=er_ws_step,
                wd_bin_width=er_wd_bin_width,
                plot_iter_path=plot_iter_path,
                fast=True,
            )

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
            finish=opt_finish,
            # workers=opt_workers,
        )

        wd_bias = x_opt
        self.opt_wd_bias = wd_bias
        self.opt_cost = J_opt
        self.opt_wd_grid = x
        self.opt_wd_cost = J

        # End with optimal results and bootstrapping
        print('  Evaluating optimal solution with bootstrapping')
        self._get_energy_ratios_allbins(
            wd_bias=x_opt,
            time_mask=time_mask,
            ws_mask=ws_mask,
            wd_mask=wd_mask,
            ti_mask=ti_mask,
            wd_step=er_wd_step,
            ws_step=er_ws_step,
            wd_bin_width=er_wd_bin_width,
            N_btstrp=er_N_btstrp,
            plot_iter_path=None,
            fast=False
        )

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
        fig_list = []
        ax_list = []
        for ii, fsc in enumerate(self.fsc_list):
            ti = self.fsc_test_turbine_list[ii]
            ax = fsc.plot_energy_ratios()
            ax[0].set_title('Turbine {:03d}'.format(ti))
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path + '_{:03d}.{:s}'.format(ti, format), dpi=dpi)

            fig_list.append(plt.gcf())
            ax_list.append(ax)

        return fig_list, ax_list
