"""Module to estimate the wind direction bias."""

from __future__ import annotations

import os as os
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floris.utilities import wrap_360
from scipy import optimize as opt, stats as spst

from flasc import FlascDataFrame
from flasc.analysis import energy_ratio as er
from flasc.analysis.energy_ratio_input import EnergyRatioInput
from flasc.data_processing import dataframe_manipulations as dfm
from flasc.logging_manager import LoggingManager
from flasc.utilities import floris_tools as ftools


class bias_estimation(LoggingManager):
    """Class to determine bias in wind direction measurement.

    This class can be used to estimate the bias (offset) in a wind
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
        df: Union[pd.DataFrame, FlascDataFrame],
        df_fm_approx: pd.DataFrame,
        test_turbines_subset: List[int],
        df_ws_mapping_func: Callable,
        df_pow_ref_mapping_func: Callable,
    ):
        """Initialize the bias estimation class.

        Args:
            df (pd.Dataframe | FlascDataFrame): Dataframe with the SCADA data measurements
                formatted in the generic format. The dataframe should contain
                at the minimum the following columns:
                * Reference wind direction for the test turbine, 'wd'
                * Reference wind speed for the test turbine, 'ws'
                * Power production of every turbine: pow_000, pow_001, ...
                * Reference power production used to normalize the energy
                    ratio: 'pow_ref'
            df_fm_approx (pd.Dataframe): Dataframe containing a large set
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
            df_pow_ref_mapping_func ([function]): This is a function that
                returns the reference power production based on an array of
                wind directions as input.
        """
        self.logger.info("Initializing a bias_estimation() object...")

        # Import inputs
        self.df = df.reset_index(drop=("time" in df.columns))
        self.n_turbines = dfm.get_num_turbines(self.df)
        self.df_fm_approx = df_fm_approx
        self.df_ws_mapping_func = df_ws_mapping_func
        self.df_pow_ref_mapping_func = df_pow_ref_mapping_func
        self.test_turbines = test_turbines_subset

    # Private methods

    def _load_er_input_for_wd_bias(
        self,
        wd_bias,
    ):
        """Load EnergyRatioInput objects with bias.

        This function initializes an instance of the EnergyRatioInput
        where the dataframe is shifted by wd_bias for each test turbine.
        This facilitates the calculation of the energy ratios under this
        hypothesized wind direction bias. Additionally, the FLORIS predictions
        for the shifted dataset are calculated and the energy ratios for the FLORIS
        predictions are also calculated.

        Args:
            wd_bias (float): Hypothesized wind direction bias in degrees.
            test_turbines ([iteratible]): List of test turbines for
                which each the energy ratios are calculated and the Pearson
                correlation coefficients are calculated. Note that this
                variable is slightly different from 'test_turbines' in the
                energy ratio classes. Namely, in this class, the energy ratio
                is calculated for each entry in test_turbines.

        Returns:
            et ([polars dataframe]): The energy ratio table
                object in which the inserted dataframe has a shifted
                wind direction measurement, offset by 'wd_bias' compared
                to the nominal dataset.
        """
        self.logger.info("  Constructing energy table for wd_bias of %.2f deg." % wd_bias)

        er_in_test_turbine_list_scada = []
        er_in_test_turbine_list_floris = []

        # Derive dataframe that covers all test_turbines
        df_cor_all = self.df.copy()
        df_cor_all["wd"] = wrap_360(df_cor_all["wd"] - wd_bias)

        # Set columns 'ws' and 'pow_ref' for df_subset_cor
        df_cor_all = self.df_ws_mapping_func(df_cor_all)
        df_cor_all = self.df_pow_ref_mapping_func(df_cor_all)
        df_cor_all = df_cor_all.dropna(subset=["wd", "ws", "pow_ref"])
        df_cor_all = df_cor_all.reset_index(drop=True)

        # Get FLORIS predictions
        self.logger.info("    Interpolating FLORIS predictions for dataframe.")
        ws_cols = ["ws_{:03d}".format(ti) for ti in range(self.n_turbines)]
        pow_cols = ["pow_{:03d}".format(ti) for ti in range(self.n_turbines)]
        df_fm_all = df_cor_all[["time", "wd", "ws", "ti", *ws_cols, *pow_cols]].copy()

        df_fm_all = ftools.interpolate_floris_from_df_approx(
            df=df_fm_all, df_approx=self.df_fm_approx, verbose=False, mirror_nans=True
        )
        df_fm_all = self.df_pow_ref_mapping_func(df_fm_all)

        for ti in self.test_turbines:
            valid_entries = (~df_cor_all["pow_{:03d}".format(ti)].isna()) & (
                ~df_fm_all["pow_{:03d}".format(ti)].isna()
            )
            df_cor = df_cor_all[valid_entries].copy().reset_index(drop=True)
            df_fm = df_fm_all[valid_entries].copy().reset_index(drop=True)

            # Initialize SCADA analysis class and add dataframes
            er_in_test_turbine_list_scada.append(EnergyRatioInput([df_cor], ["Measured data"]))
            er_in_test_turbine_list_floris.append(EnergyRatioInput([df_fm], ["FLORIS prediction"]))

        # Save to self
        self.er_in_test_turbine_list_scada = er_in_test_turbine_list_scada
        self.er_in_test_turbine_list_floris = er_in_test_turbine_list_floris

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
    ):
        """Calculate the energy ratios.

        Args:
            wd_bias (float): Wind direction bias in degrees.
            time_mask ([iterable], optional): Mask.  If None, will not mask
                the data based on this variable. Defaults to None.
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
        er_out_test_turbine_list_scada = []
        er_out_test_turbine_list_floris = []

        if time_mask is not None:
            raise NotImplementedError(
                "time_mask not available. Please preprocess " + "your dataset to apply time masks."
            )
        if ti_mask is not None:
            raise NotImplementedError(
                "ti_mask not available. Please preprocess "
                + "your dataset to apply turbulence intensity masks."
            )
        if wd_mask is None:
            wd_mask = [0.0, 360.0]

        self.logger.info("    Initializing energy ratio inputs.")
        self._load_er_input_for_wd_bias(wd_bias=wd_bias)

        for ii, ti in enumerate(self.test_turbines):
            self.logger.info(
                "    Determining energy ratios for test turbine = %03d." % (ti)
                + " WD bias: %.3f deg." % wd_bias
            )

            er_out_test_turbine_list_scada.append(
                er.compute_energy_ratio(
                    self.er_in_test_turbine_list_scada[ii],
                    ref_turbines=None,
                    test_turbines=[ti],
                    use_predefined_ref=True,
                    use_predefined_wd=True,
                    use_predefined_ws=True,
                    wd_step=wd_step,
                    wd_min=wd_mask[0],
                    wd_max=wd_mask[1],
                    ws_step=ws_step,
                    ws_min=ws_mask[0],
                    ws_max=ws_mask[1],
                    wd_bin_overlap_radius=(wd_bin_width - wd_step) / 2,
                    N=N_btstrp,
                )
            )

            er_out_test_turbine_list_floris.append(
                er.compute_energy_ratio(
                    self.er_in_test_turbine_list_floris[ii],
                    ref_turbines=None,
                    test_turbines=[ti],
                    use_predefined_ref=True,
                    use_predefined_wd=True,
                    use_predefined_ws=True,
                    wd_step=wd_step,
                    wd_min=wd_mask[0],
                    wd_max=wd_mask[1],
                    ws_step=ws_step,
                    ws_min=ws_mask[0],
                    ws_max=ws_mask[1],
                    wd_bin_overlap_radius=(wd_bin_width - wd_step) / 2,
                    N=N_btstrp,
                )
            )

        # Debugging: plot iteration to path
        if plot_iter_path is not None:
            self.logger.info("    Plotting energy ratios and saving figures")
            fp = os.path.join(
                plot_iter_path,
                "bias%+.3f" % (self.fsc_wd_bias_list[ii]),
                "energy_ratios_test_turbine",
            )
            self.plot_energy_ratios(save_path=fp, format="png", dpi=200)
            plt.close("all")

        # Save to self
        self.er_out_test_turbine_list_scada = er_out_test_turbine_list_scada
        self.er_out_test_turbine_list_floris = er_out_test_turbine_list_floris

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
        """Calculate Baseline energy ratios.

        Args:
            time_mask ([iterable], optional): Time Mask.
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
            er_wd_step (float, optional): Wind direction step size.  Defaults to 3.0.
            er_ws_step (float, optional): Wind speed step size. Defaults to 5.0.
            er_wd_bin_width ([type], optional): Wind direction bin width.  Defaults to None.
            er_N_btstrp (int, optional): Number of bootstrap evaluations for
                uncertainty quantification (UQ). If N_btstrp=1, will not
                perform any uncertainty quantification. Defaults to 1.
        """
        # TODO: is this calculate_baseline method needed?
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
        """Estimate wd bias.

        Estimate the wind direction bias by comparing the SCADA data
        under various wind direction corrections to its FLORIS predictions.

        Args:
            time_mask ([iterable], optional): Time mask. Should be an
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
            opt_workers (int, optional): Number of workers to use for the
                optimization. Defaults to 4.
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
            A tuple (float, float): The optimal wind direction offset and
                the cost function under the optimal offset.
        """
        self.logger.info("Estimating the wind direction bias")

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
                N_btstrp=1,
                plot_iter_path=plot_iter_path,
            )

            # Calculate cost
            cost_array = np.full(len(self.er_out_test_turbine_list_scada), np.nan)
            for ii, _ in enumerate(self.test_turbines):
                y_scada = np.array(
                    self.er_out_test_turbine_list_scada[ii].df_result["Measured data"]
                )
                y_floris = np.array(
                    self.er_out_test_turbine_list_floris[ii].df_result["FLORIS prediction"]
                )
                ids = ~np.isnan(y_scada) & ~np.isnan(y_floris)
                if np.sum(ids) > 5:  # At least 6 valid data entries
                    r, _ = spst.pearsonr(y_scada[ids], y_floris[ids])
                else:
                    r = np.nan
                cost_array[ii] = -1.0 * r

            cost = np.nanmean(cost_array)
            return cost

        def opt_finish(func, x0, args=()):
            return opt.fmin(func, x0, args, maxfun=10, full_output=True, xtol=0.1, disp=True)

        dran = opt_search_range[1] - opt_search_range[0]
        x_opt, J_opt, x, J = opt.brute(
            func=cost_fun,
            ranges=[opt_search_range],
            Ns=int(np.ceil(dran / opt_search_brute_dx) + 1),
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
        self.logger.info("  Evaluating optimal solution with bootstrapping")
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
        )

        # Save input arguments for future use
        self._input_args = {
            "time_mask": time_mask,
            "ws_mask": ws_mask,
            "wd_mask": wd_mask,
            "ti_mask": ti_mask,
            "opt_search_range": opt_search_range,
            "opt_search_brute_dx": opt_search_brute_dx,
            "opt_workers": opt_workers,
            "er_wd_step": er_wd_step,
            "er_ws_step": er_ws_step,
            "er_wd_bin_width": er_wd_bin_width,
            "er_N_btstrp": er_N_btstrp,
            "plot_iter_path": plot_iter_path,
        }

        return x_opt, J_opt

    def plot_energy_ratios(
        self, show_uncorrected_data=False, save_path=None, format="png", dpi=200
    ):
        """Plot energy ratios.

        Plot the energy ratios for the currently evaluated wind
        direction offset term.

        Args:
            show_uncorrected_data (bool, optional): Compute and show the
                uncorrected energy ratio (with wd_bias=0) on the plot. Defaults
                to False.
            save_path ([str], optional): Path to save the figure to. If not
                specified, will not save the figure. Defaults to None.
            format (str, optional): Figure format. Defaults to 'png'.
            dpi (int, optional): Figure DPI. Defaults to 200.
        """
        fig_list = []
        ax_list = []
        if show_uncorrected_data:
            # Store existing scada result
            er_out_test_turbine_list_scada_copy = self.er_out_test_turbine_list_scada.copy()
            # (Re)compute case with wd_bias=0
            self._get_energy_ratios_allbins(
                wd_bias=0,
                time_mask=self._input_args["time_mask"],
                ws_mask=self._input_args["ws_mask"],
                wd_mask=self._input_args["wd_mask"],
                ti_mask=self._input_args["ti_mask"],
                wd_step=self._input_args["er_wd_step"],
                ws_step=self._input_args["er_ws_step"],
                wd_bin_width=self._input_args["er_wd_bin_width"],
                N_btstrp=self._input_args["er_N_btstrp"],  # What should go here?
                plot_iter_path=None,
            )

            er_out_test_turbine_list_scada_0bias = self.er_out_test_turbine_list_scada.copy()

            self.er_out_test_turbine_list_scada = er_out_test_turbine_list_scada_copy

        # Plot
        for ii, ti in enumerate(self.test_turbines):
            if show_uncorrected_data:
                axarr = er_out_test_turbine_list_scada_0bias[ii].plot_energy_ratios(
                    labels=["Measured data (uncorrected)"],
                    color_dict={"Measured data (uncorrected)": "silver"},
                    show_wind_speed_distribution=False,
                )
                axarr = self.er_out_test_turbine_list_scada[ii].plot_energy_ratios(
                    labels=["Measured data (bias corrected)"],
                    color_dict={"Measured data (bias corrected)": "C0"},
                    axarr=axarr,
                    show_wind_speed_distribution=False,
                )
            else:
                axarr = self.er_out_test_turbine_list_scada[ii].plot_energy_ratios(
                    color_dict={"Measured data": "C0"}, show_wind_speed_distribution=False
                )

            axarr = self.er_out_test_turbine_list_floris[ii].plot_energy_ratios(
                color_dict={"FLORIS prediction": "C1"},
                axarr=axarr,
                show_wind_speed_distribution=False,
            )

            axarr[0].set_title("Turbine {:03d}".format(ti))
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path + "_{:03d}.{:s}".format(ti, format), dpi=dpi)

            fig_list.append(plt.gcf())
            ax_list.append(axarr)

        return fig_list, ax_list
