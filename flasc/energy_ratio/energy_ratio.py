# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
import pandas as pd
from random import choices

from floris.utilities import wrap_360
from pandas.errors import DataError

from ..dataframe_operations import dataframe_manipulations as dfm
from ..energy_ratio import energy_ratio_visualization as ervis


class energy_ratio:
    """This class is used to calculate the energy ratios for a single
    dataframe with measurements, either from FLORIS or from SCADA data.
    This class supports bootstrapping for uncertainty quantification,
    automatic derivation of the frequency of bins based on occurrence
    in the provided dataset, and various choices for binning and daa
    discretization.
    """

    def __init__(self, df_in, inflow_freq_interpolant=None, verbose=False):
        """Initialization of the class.

        Args:
            df_in ([pd.DataFrame]): The dataframe provided by the user. This
            dataframe should have the following columns:
                * Reference wind direction for the test turbine, 'wd'
                * Reference wind speed for the test turbine, 'ws'
                * Power production of every turbine: pow_000, pow_001, ...
                * Reference power production used to normalize the energy
                    ratio: 'pow_ref'
            inflow_freq_interpolant (interpolant, optional): This is an
            interpolant that takes as inputs the wind direction and wind
            speed, and then returns the frequency of occurrence for that set
            of inflow conditions. If None is specified, the occurrence of each
            bin is derived from the provided data, df_in. Defaults to None.
            verbose (bool, optional): Print to console. Defaults to False.
        """
        self.verbose = verbose

        # Initialize dataframe
        self._set_df(df_in)

        # Initialize frequency functions
        self._set_inflow_freq_interpolant(inflow_freq_interpolant)

    # Private methods

    def _set_inflow_freq_interpolant(self, inflow_freq_interpolant):
        self.inflow_freq_interpolant = inflow_freq_interpolant

    def _set_df(self, df_in):
        """This function writes the dataframe provided by the user to the
        class as self.df_full. This full dataframe will be used to create
        a minimal dataframe called self.df which contains the minimum
        columns required to calculate the energy ratios. The contents of
        self.df will depend on the test_turbine specified and hence
        that dataframe is created in the _set_test_turbines() function.

        Args:
            df_in ([pd.DataFrame]): The dataframe provided by the user. This
            dataframe should have the following columns:
                * Reference wind direction for the test turbine, 'wd'
                * Reference wind speed for the test turbine, 'ws'
                * Power production of every turbine: pow_000, pow_001, ...
                * Reference power production used to normalize the energy
                    ratio: 'pow_ref'
        """
        if "pow_ref" not in df_in.columns:
            raise KeyError("pow_ref column not in dataframe. Cannot proceed.")
            # INFO: You can add such a column using:
            #   from flasc.dataframe_operations import \
            #       dataframe_manipulations as dfm
            #
            #   df = dfm.set_pow_ref_by_*(df)
            #   ...

        # Copy full dataframe to self
        self.df_full = df_in.copy()  # Full dataframe
        self.df = None

    def _set_test_turbines(self, test_turbines):
        """This function calculates the power production upon which the
        energy ratio is calculated, in the nominator of the energy ratio
        equation. This is typically a single turbine, e.g.,
        test_turbines=[0], but can also be the average of multiple turbines,
        e.g., test_turbines=[0, 1, 2]. This function creates the minimal
        dataframe, self.df, with columns being the wind direction 'wd',
        the wind speed 'ws', the power production of the test turbine(s)
        'pow_test' and the reference power production 'pow_ref'. The
        arrays 'pow_test' and 'pow_ref' are in the nominator and
        denominator in the energy ratio equation, respectively.

        Args:
            test_turbines ([iteratible]): List with the test turbine(s)
                used to calculate the power production in the nominator of
                the energy ratio equation. Typically, this is a single
                turbine, e.g., test_turbines=[0], but can also be multiple
                turbines. If multiple turbines are specified, it averages
                the power production between the turbines to come up with
                the test power values.
        """
        if not (type(test_turbines) is list):
            test_turbines = [test_turbines]
        self.test_turbines = test_turbines

        if "ti" in self.df_full.columns:
            cols = ["wd", "ws", "ti", "pow_ref"]
        else:
            cols = ["wd", "ws", "pow_ref"]

        self.df = self.df_full[cols].copy()
        self.df["pow_test"] = dfm.get_column_mean(
            df=self.df_full,
            col_prefix="pow",
            turbine_list=self.test_turbines,
            circular_mean=False,
        )

    def _set_binning_properties(
        self, ws_step=None, wd_step=None, wd_bin_width=None,
        ws_bins=None, wd_bins=None, 
    ):
        """This function prepares the wind direction and wind speed bins in
        accordance to the user specified functions. Previously, the user could
        only specify the bins by assigning a ws_step and wd_step. Now, you can
        also specify the bins by assigning them directly. If 'ws_bins' is
        provided, then the variable 'ws_step' is completely ignored and the
        wind speed bins are directly set as the user-provided values in
        'ws_bins'. The same holds for wd_bins. If 'wd_bins' is provided, the
        variables 'wd_step' and 'wd_bin_width' are ignored and the wind
        direction bins are directly assigned the values in 'wd_bins'.

        Args:
            ws_step (float): Wind speed bin width and defines the step size
                at which the energy ratios are calculated along the wind speed.
                If 'ws_bins' is also provided, this variable is ignored.
            wd_step (float): Wind direction bin width and defines the step
                size at which the energy ratios are calculated along the wind
                direction. If 'wd_bins' is also provided, this variable is
                ignored.
            wd_bin_width (float, optional): Width of each wind direction bin.
                If this is larger than wd_step, there is overlap in the energy
                ratios between bins. This means data points are used more than
                once -- i.e., fall into multiple bins at the same time. If None
                is specified, defaults to the same value as wd_step. Note that
                if 'wd_bins' is also provided, this variable is ignored. Defaults
                to None.
            ws_bins (array, optional): Array containing the bins over which
                the energy ratios must be calculated (wind speeds). Each entry
                of the provided array must contain exactly two float values,
                being the lower and upper bound for that wind speed bin.
                Overlap between bins is not supported for wind speed bins,
                currently. Defaults to None.
            wd_bins (array, optional): Array containing the bins over which
                the energy ratios must be calculated (wind dir.). Each entry
                of the provided array must contain exactly two float values,
                being the lower and upper bound for that wind dir. bin.
                Overlap between bins is supported for wind direction bins.
                Defaults to None.
        """
        if ((ws_bins is None) | (wd_bins is None)):
            # Add a temporary variable
            a = np.array([-0.5, 0.5], dtype=float)

        # If ws_bins is not specified, automatically calculate the
        # wind speed bins using ws_step as bin width, from ws_step / 2.0,
        # which is bounded by [0.0, ws_step),  up to 30 m/s. If the user
        # already provided the wind speed bins 'ws_bins', then we need not
        # derive anything and we directly save the user-specified bins
        # to self (after converting them to numpy arrays).
        if wd_bins is None:
            ws_step = float(ws_step)
            ws_labels = np.arange(ws_step/2.0, 30.0001, ws_step)
            ws_bins = np.array([ws + a * ws_step for ws in ws_labels])
        else:
            ws_labels = np.array([np.mean(b) for b in ws_bins], dtype=float)
            ws_bins = np.array(ws_bins, dtype=float)

        # If wd_bins is not specified, automatically calculate the wind
        # direction bins using wd_step as bin width, from 0.0 deg to 360 deg.
        # If the user has already provided the wind direction bins 'ws_bins',
        # then we need not derive anything and we directly save the user
        # specified bins to self (after converting them to numpy arrays).
        if wd_bins is None:
            wd_step = float(wd_step)
            if wd_bin_width is None:
                wd_bin_width = wd_step
            wd_bin_width = float(wd_bin_width)

            wd_min = np.min([wd_step / 2.0, wd_bin_width / 2.0])
            wd_labels = np.arange(wd_min, 360.0001, wd_step)
            wd_bins = np.array([wd + a * wd_bin_width for wd in wd_labels])
        else:
            wd_labels = np.array([np.mean(b) for b in wd_bins], dtype=float)
            wd_bins = np.array(wd_bins, dtype=float)

        # Save variables
        self.ws_step = ws_step
        self.wd_step = wd_step
        self.wd_bin_width = wd_bin_width
        self.ws_labels = ws_labels
        self.wd_labels = wd_labels
        self.ws_bins = ws_bins
        self.wd_bins = wd_bins

    def _calculate_bins(self):
        """This function bins the data in the minimal dataframe, self.df,
        into the respective wind direction and wind speed bins. Note that
        there might be bin overlap if the specified wd_bin_width is larger
        than the bin step size. This code will copy dataframe rows that fall
        into multiple bins, effectively increasing the sample size.
        """
        # Bin according to wind speed. Note that data never falls into
        # multiple wind speed bins at the same time.
        for ws_bin in self.ws_bins:
            ws_interval = pd.Interval(ws_bin[0], ws_bin[1], "left")
            ids = (self.df["ws"] >= ws_bin[0]) & (self.df["ws"] < ws_bin[1])
            self.df.loc[ids, "ws_bin"] = np.mean(ws_bin)
            self.df.loc[ids, "ws_bin_edges"] = ws_interval

        # Bin according to wind direction. Note that data can fall into
        # multiple wind direction bins at the same time, if wd_bin_width is
        # larger than the wind direction binning step size, wd_step. If so,
        # data will be copied and the sample size is effectively increased
        # so that every relevant bin has that particular measurement.
        df_list = [None for _ in range(len(self.wd_labels))]
        for ii, wd_bin in enumerate(self.wd_bins):
            wd_interval = pd.Interval(wd_bin[0], wd_bin[1], "left")
            lb = wrap_360(wd_bin[0])
            ub = wrap_360(wd_bin[1])
            if ub < lb:  # Deal with angle wrapping
                ids = (self.df["wd"] >= lb) | (self.df["wd"] < ub)
            else:
                ids = (self.df["wd"] >= lb) & (self.df["wd"] < ub)
            df_subset = self.df.loc[ids].copy()
            df_subset["wd_bin"] = np.mean(wd_bin)
            df_subset["wd_bin_edges"] = wd_interval
            df_list[ii] = df_subset
        self.df = pd.concat(df_list, copy=False)

        # Make sure a float
        self.df["ws_bin"] = self.df["ws_bin"].astype(float)
        self.df["wd_bin"] = self.df["wd_bin"].astype(float)

    def _get_df_freq(self):
        """This function derives the frequency of occurrence of each bin
        (wind direction and wind speed) from the binned dataframe. The
        found values are used in the energy ratio equation to weigh the
        power productions of each bin according to their frequency of
        occurrence.
        """
        # Determine observed frequency
        cols = ["ws_bin", "wd_bin", "ws_bin_edges", "wd_bin_edges"]
        df_freq_observed = self.df[cols].copy()
        df_freq_observed["freq"] = 1
        df_freq_observed = df_freq_observed.groupby(["wd_bin", "ws_bin"])
        bin_edges = df_freq_observed[["ws_bin_edges", "wd_bin_edges"]].first()
        bin_freq = df_freq_observed["freq"].sum()
        df_freq_observed = pd.concat([bin_freq, bin_edges], axis=1)
        df_freq_observed = df_freq_observed.reset_index(drop=False)
        df_freq = df_freq_observed

        if self.inflow_freq_interpolant is not None:
            # Overwrite freq of bin occurrence with user-specified function
            df_freq["freq"] = self.inflow_freq_interpolant(
                df_freq["wd_bin"],
                df_freq["ws_bin"],
            )

        # Sort by 'ws_bin' as index
        df_freq = df_freq.set_index("ws_bin")
        self.df_freq = df_freq

        return df_freq

    # Public methods

    def get_energy_ratio(
        self,
        test_turbines,
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=None,
        wd_bins=None,
        ws_bins=None,
        N=1,
        percentiles=[5.0, 95.0],
        return_detailed_output=False,
        num_blocks=-1
    ):
        """This is the main function used to calculate the energy ratios
        for dataframe provided to the class during initialization. One
        can calculate the energy ratio for different (sets of) turbines
        and under various discretization options.

        Args:
            test_turbines ([iteratible]): List with the test turbine(s)
                used to calculate the power production in the nominator of
                the energy ratio equation. Typically, this is a single
                turbine, e.g., test_turbines=[0], but can also be multiple
                turbines. If multiple turbines are specified, it averages
                the power production between the turbines to come up with
                the test power values.
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
            ws_bins (array, optional): Array containing the bins over which
                the energy ratios must be calculated (wind speeds). Each entry
                of the provided array must contain exactly two float values,
                being the lower and upper bound for that wind speed bin.
                Overlap between bins is not supported for wind speed bins,
                currently. Defaults to None.
            wd_bins (array, optional): Array containing the bins over which
                the energy ratios must be calculated (wind dir.). Each entry
                of the provided array must contain exactly two float values,
                being the lower and upper bound for that wind dir. bin.
                Overlap between bins is supported for wind direction bins.
                Defaults to None.
            N (int, optional): Number of bootstrap evaluations for
                uncertainty quantification (UQ). If N=1, will not perform
                any uncertainty quantification. Defaults to 1.
            percentiles (list, optional): Confidence bounds for the
                uncertainty quantification in percents. This value is only
                relevant if N > 1 is specified. Defaults to [5., 95.].
            return_detailed_output (bool, optional): Also calculate and
                return detailed energy ratio information useful for debugging
                and figuring out flaws in the data. This slows down the
                calculations but can be very useful. The additional info is
                written to self.df_lists[i]["er_results_info_dict"]. The
                dictionary variable therein contains two fields, being
                "df_per_wd_bin" and "df_per_ws_bin". The first gives an
                overview of the energy ratio for every wind direction bin,
                covering the collective effect of all wind speeds in the
                data. The latter one, "df_per_ws_bin", yields even more
                information and displays the energy ratio for every wind
                direction and wind speed bin, among others. This is
                particularly helpful in figuring out if the bins are well
                balanced. Defaults to False.
            num_blocks (int, optional): Number of blocks to use in block
                boostrapping.  If = -1 then don't use block bootstrapping
                and follow normal approach of sampling num_samples randomly
                with replacement.  Defaults to -1.

        Returns:
            energy_ratios ([pd.DataFrame]): Dataframe containing the found
                energy ratios under the prespecified settings. The dataframe
                contains the columns:
                    * wd_bin: The mean wind direction for this bin
                    * N_bin: Number of data entries in this bin
                    * baseline: Nominal energy ratio value (without UQ)
                    * baseline_l: Lower bound for energy ratio. This
                        value is equal to baseline without UQ and lower
                        with UQ.
                    * baseline_u: Upper bound for energy ratio. This
                        value is equal to baseline without UQ and higher
                        with UQ.
        """
        if self.df_full.shape[0] < 1:
            # Empty dataframe, do nothing
            self.energy_ratio_out = pd.DataFrame()
            self.energy_ratio_N = N
            return None

        if self.verbose:
            print("Calculating energy ratios with N = %d." % N)

        # Set up a 'pow_test' column in the dataframe
        self._set_test_turbines(test_turbines)

        # Set up bins
        self._set_binning_properties(
            ws_step=ws_step, wd_step=wd_step, wd_bin_width=wd_bin_width,
            ws_bins=ws_bins, wd_bins=wd_bins
        )
        self._calculate_bins()

        # Get probability distribution of bins
        self._get_df_freq()

        # Calculate the energy ratio for all bins
        out = _get_energy_ratios_all_wd_bins_bootstrapping(
            df_binned=self.df,
            df_freq=self.df_freq,
            N=N,
            percentiles=percentiles,
            return_detailed_output=return_detailed_output,
            num_blocks=num_blocks
        )
        if return_detailed_output:
            energy_ratios = out[0]
            dict_out = out[1]
        else:
            energy_ratios = out

        self.energy_ratio_out = energy_ratios
        self.energy_ratio_N = N

        if return_detailed_output:
            return energy_ratios, dict_out

        return energy_ratios

    def get_energy_ratio_fast(
        self, test_turbines, ws_step, wd_step, wd_bin_width=None, 
        ws_bins=None, wd_bins=None,
    ):
        """This function calculates the energy ratio in a fast manner
        but completely ignores any frequency weighing to achieve these speed-
        ups. It also does not support bootstrapping because of this.

        Args:
            test_turbines ([iteratible]): List with the test turbine(s)
                used to calculate the power production in the nominator of
                the energy ratio equation. Typically, this is a single
                turbine, e.g., test_turbines=[0], but can also be multiple
                turbines. If multiple turbines are specified, it averages
                the power production between the turbines to come up with
                the test power values.
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

        Returns:
            [type]: [description]
        """
        # Choose default option
        if wd_bin_width is None:
            wd_bin_width = wd_step

        # Set up a 'pow_test' column in the dataframe
        self._set_test_turbines(test_turbines)

        # Set up bins
        self._set_binning_properties(
            ws_step, wd_step, wd_bin_width, ws_bins, wd_bins
        )
        self._calculate_bins()

        df = self.df.dropna(how="any", subset=["pow_ref", "pow_test"]).copy()
        df.loc[df.index, "bin_count"] = 1

        df_summed = df.groupby("wd_bin")[["pow_ref", "pow_test", "bin_count"]].sum()
        df_summed.loc[df_summed.index, ["baseline", "baseline_lb", "baseline_ub"]] = (
            np.tile(df_summed["pow_test"] / df_summed["pow_ref"], (3, 1)).T
        )
        energy_ratios = df_summed.reset_index(drop=False)
        return energy_ratios

    def plot_energy_ratio(self, hide_uq_labels=True, polar_plot=False):
        """This function plots the energy ratio against the wind direction,
        potentially with uncertainty bounds if N > 1 was specified by
        the user. One must first run get_energy_ratio() before attempting
        to plot the energy ratios.

        Args:
            hide_uq_labels (bool, optional): If true, do not specifically label
                the confidence intervals in the plot
            polar_plot (bool, optional): Plots the energy ratios in a polar
                coordinate system, aligned with the wind direction coordinate
                system of FLORIS. Defaults to False.

        Returns:
            ax [plt.Axes]: Axis handle for the figure.
        """
        return ervis.plot(
            self.energy_ratio_out,
            self.df_freq.reset_index(drop=False),
            hide_uq_labels=hide_uq_labels,
            polar_plot=polar_plot
        )


# Support functions not included in energy_ratio class


def _get_energy_ratios_all_wd_bins_bootstrapping(
    df_binned,
    df_freq,
    N=1,
    percentiles=[5.0, 95.0],
    return_detailed_output=False,
    num_blocks=-1
):
    """Wrapper function that calculates the energy ratio for every wind
    direction bin in the provided dataframe. This function wraps around
    the function '_get_energy_ratio_single_wd_bin_bootstrapping', which
    calculates the energy ratio for a single wind direction bin.

    Args:
        df_binned ([pd.DataFrame]): Dataframe containing the binned
        data. This dataframe must contain, at the minimum, the following
        columns:
            * ws_bin: The wind speed bin
            * wd_bin: The wind direction bin
            * pow_ref: The reference power production, previously specified
                by the user outside of this function/class. This value
                belongs in the denominator in the energy ratio equation.
            * pow_test: The test power production. This value belongs in the
                nominator in the energy ratio equation.
        df_freq ([pd.DataFrame]): Dataframe containing the frequency of every
            wind direction and wind speed bin. This dataframe is typically
            derived from the data itself but can also be a separate dataframe
            based on the wind rose of the site.
        N (int, optional): Number of bootstrap evaluations for
            uncertainty quantification (UQ). If N=1, will not perform any
            uncertainty quantification. Defaults to 1.
        percentiles (list, optional): Confidence bounds for the
            uncertainty quantification in percents. This value is only
            relevant if N > 1 is specified. Defaults to [5., 95.].
        return_detailed_output (bool, optional): Also calculate and
            return detailed energy ratio information useful for debugging
            and figuring out flaws in the data. This slows down the
            calculations but can be very useful. The additional info is
            written to self.df_lists[i]["er_results_info_dict"]. The
            dictionary variable therein contains two fields, being
            "df_per_wd_bin" and "df_per_ws_bin". The first gives an
            overview of the energy ratio for every wind direction bin,
            covering the collective effect of all wind speeds in the
            data. The latter one, "df_per_ws_bin", yields even more
            information and displays the energy ratio for every wind
            direction and wind speed bin, among others. This is
            particularly helpful in figuring out if the bins are well
            balanced. Defaults to False.
        num_blocks (int, optional): Number of blocks to use in block
            boostrapping.  If = -1 then don't use block bootstrapping
            and follow normal approach of sampling num_samples randomly
            with replacement.  Defaults to -1.

    Returns:
        energy_ratios ([pd.DataFrame]): Dataframe containing the found
            energy ratios under the prespecified settings. The dataframe
            contains the columns:
                * wd_bin: The mean wind direction for this bin
                * N_bin: Number of data entries in this bin
                * baseline: Nominal energy ratio value (without UQ)
                * baseline_l: Lower bound for energy ratio. This
                    value is equal to baseline without UQ and lower
                    with UQ.
                * baseline_u: Upper bound for energy ratio. This
                    value is equal to baseline without UQ and higher
                    with UQ.
    """
    # Extract minimal dataframe
    if "ti" in df_binned.columns:
        min_cols = [
            "wd",
            "ws",
            "ti",
            "ws_bin",
            "wd_bin",
            "pow_ref",
            "pow_test",
        ]
    else:
        min_cols = ["wd", "ws", "ws_bin", "wd_bin", "pow_ref", "pow_test"]
    df = df_binned[min_cols]

    # Save some relevant info
    unique_wd_bins = np.unique(df.wd_bin)
    # unique_ws_bins = np.unique(df.ws_bin)

    # Now calculate the actual energy ratios
    result = np.zeros([len(unique_wd_bins), 3])
    dict_out_list = [None for _ in range(len(unique_wd_bins))]

    for wd_idx, wd in enumerate(unique_wd_bins):
        df_subset = df[df["wd_bin"] == wd]
        df_freq_subset = df_freq[df_freq["wd_bin"] == wd]

        out = _get_energy_ratio_single_wd_bin_bootstrapping(
                df_binned=df_subset,
                df_freq=df_freq_subset,
                N=N,
                percentiles=percentiles,
                return_detailed_output=return_detailed_output,
                num_blocks = num_blocks
        )
        if return_detailed_output:
            result[wd_idx, :] = out[0]
            dict_out_list[wd_idx] = out[1]
        else:
            result[wd_idx, :] = out

    # Save energy ratios to the dataframe
    df_out = pd.DataFrame(
        result, columns=["baseline", "baseline_lb", "baseline_ub"]
    )

    # Save wind direction bins and bin count to dataframe
    df_out["wd_bin"] = unique_wd_bins
    _, df_out["bin_count"] = np.unique(df["wd_bin"], return_counts=True)
    df_out["bin_count"] = df_out["bin_count"].astype(int)

    if return_detailed_output:
        # Concatenate dataframes and produce a new dict_out
        df_per_wd_bin = pd.concat([d["df_per_wd_bin"] for d in dict_out_list])
        df_per_ws_bin = pd.concat([d["df_per_ws_bin"] for d in dict_out_list])
        df_per_ws_bin = df_per_ws_bin.reset_index(drop=False)
        df_per_ws_bin = df_per_ws_bin.set_index(["wd_bin"])
        dict_out = {
            "df_per_wd_bin": df_per_wd_bin,
            "df_per_ws_bin": df_per_ws_bin,
        }
        return df_out, dict_out

    return df_out


def _get_energy_ratio_single_wd_bin_bootstrapping(
    df_binned,
    df_freq,
    N=1,
    percentiles=[5.0, 95.0],
    return_detailed_output=False,
    num_blocks = -1
):
    """Get the energy ratio for one particular wind direction bin and
    an array of wind speed bins. This function also includes bootstrapping
    functionality by increasing the number of bootstrap evaluations (N) to
    larger than 1. The bootstrap percentiles default to 5 % and 95 %.

    Args:
        df_binned ([pd.DataFrame]): Dataframe containing the binned
        data. This dataframe must contain, at the minimum, the following
        columns:
            * ws_bin: The wind speed bin
            * wd_bin: The wind direction bin
            * pow_ref: The reference power production, previously specified
                by the user outside of this function/class. This value
                belongs in the denominator in the energy ratio equation.
            * pow_test: The test power production. This value belongs in the
                nominator in the energy ratio equation.
        df_freq ([pd.DataFrame]): Dataframe containing the frequency of every
            wind direction and wind speed bin. This dataframe is typically
            derived from the data itself but can also be a separate dataframe
            based on the wind rose of the site.
        N (int, optional): Number of bootstrap evaluations for
            uncertainty quantification (UQ). If N=1, will not perform any
            uncertainty quantification. Defaults to 1.
        percentiles (list, optional): Confidence bounds for the
            uncertainty quantification in percents. This value is only
            relevant if N > 1 is specified. Defaults to [5., 95.].
        return_detailed_output (bool, optional): Also calculate and
            return detailed energy ratio information useful for debugging
            and figuring out flaws in the data. This slows down the
            calculations but can be very useful. The additional info is
            written to self.df_lists[i]["er_results_info_dict"]. The
            dictionary variable therein contains two fields, being
            "df_per_wd_bin" and "df_per_ws_bin". The first gives an
            overview of the energy ratio for every wind direction bin,
            covering the collective effect of all wind speeds in the
            data. The latter one, "df_per_ws_bin", yields even more
            information and displays the energy ratio for every wind
            direction and wind speed bin, among others. This is
            particularly helpful in figuring out if the bins are well
            balanced. Defaults to False.
        num_blocks (int, optional): Number of blocks to use in block
            boostrapping.  If = -1 then don't use block bootstrapping
            and follow normal approach of sampling num_samples randomly
            with replacement.  Defaults to -1.

    Returns:
        results_array ([np.array]): Numpy array of statistical results
            [nominal energy ratio, low result, high result]
            if bootstrapping disabled nominal result 3 times in row
    """
    # Get results excluding uncertainty
    if return_detailed_output:
        energy_ratio_nominal, dict_info = _get_energy_ratio_single_wd_bin_nominal(
            df_binned=df_binned,
            df_freq=df_freq,
            return_detailed_output=return_detailed_output,
        )
    else:
        energy_ratio_nominal = _get_energy_ratio_single_wd_bin_nominal(
            df_binned=df_binned,
            df_freq=df_freq,
            return_detailed_output=return_detailed_output,
        )

    # Add bootstrapping results, if necessary
    if N <= 1:
        results_array = np.array([energy_ratio_nominal] * 3, dtype=float)
    else:

        # First check, if num_blocks is > number of points, then assume normal bootstrapping
        if num_blocks > df_binned.shape[0]:
            num_blocks = -1

        # If after this revision, the number of blocks is very low, use normal bootstrapping


        # Check that num_blocks is an allowable number
        if (num_blocks < -1) or (num_blocks == 0) or (num_blocks == 1) or (num_blocks > df_binned.shape[0]):
            raise ValueError("num_blocks should either be -1 (don't use block bootstrapping) or else a number between 2 and num_samples")

        # If using block-bootstrapping, set up blocks
        if num_blocks > 0:
            block_list = list(range(num_blocks))  # List of all block names
            block_indices = np.arange(df_binned.shape[0]) # Simple index for iloc
            block_length = int(len(block_indices) / num_blocks) # Length of each block
            block_labels = np.zeros(len(block_indices)).astype(int) # Labels to assign each index
            for b_i in range(num_blocks):
                block_labels[block_length * b_i:block_length * (b_i+1)] = b_i

        # Get a bootstrap sample of range
        bootstrap_results = np.zeros(N)
        bootstrap_results[0] = energy_ratio_nominal
        for i in range(1, N):
            if num_blocks <= 0:
                df_randomized = df_binned.sample(frac=1, replace=True).copy()
            else:
                test_blocks = choices(block_list, k = num_blocks) # Choose a set of blocks
                
                # Indices of these blocks
                test_indices = []
                for b in test_blocks:
                    m = block_labels == b # Mask to find this block's indices
                    test_indices = np.append(test_indices,block_indices[m]) #Append indices

                df_randomized = df_binned.iloc[test_indices].reset_index().copy()

            bootstrap_results[i] = _get_energy_ratio_single_wd_bin_nominal(
                df_binned=df_randomized,
                df_freq=df_freq,
                return_detailed_output=False,
            )

        # Return the results in the order used in previous versions
        results_array = np.array(
            [
                energy_ratio_nominal,
                np.nanpercentile(bootstrap_results, percentiles)[0],
                np.nanpercentile(bootstrap_results, percentiles)[1],
            ]
        )

    if return_detailed_output:
        return results_array, dict_info
    else:
        return results_array


def _get_energy_ratio_single_wd_bin_nominal(
    df_binned, df_freq=None, return_detailed_output=False
):
    """Get the energy ratio for one particular wind direction bin and
    an array of wind speed bins. This function performs a single
    calculation of the energy ratios without uncertainty quantification.
    """
    # Copy minimal dataframe
    if "ti" in df_binned.columns:
        min_cols = [
            "wd_bin",
            "ws_bin",
            "wd",
            "ws",
            "ti",
            "pow_ref",
            "pow_test",
        ]
        mean_cols = ["wd", "ws", "ti", "pow_ref", "pow_test"]
        std_cols = ["wd", "ws", "ti", "pow_ref", "pow_test"]

    else:
        min_cols = ["wd", "ws", "wd_bin", "ws_bin", "pow_ref", "pow_test"]
        mean_cols = ["wd", "ws", "pow_ref", "pow_test"]
        std_cols = ["wd", "ws", "pow_ref", "pow_test"]
    df = df_binned[min_cols].copy()

    # Drop any faulty measurements
    df = df.dropna(how="any")

    # Check if only one wd_bin present in data
    wd_bin = df_binned["wd_bin"].unique()
    if len(wd_bin) > 1:
        raise DataError("More than one wd_bin present in data.")

    # Reference and test turbine energy
    df["freq"] = 1
    df_sums = df.groupby("ws_bin")[["pow_ref", "pow_test", "freq"]].sum()
    df_sums.columns = [
        "energy_ref_unbalanced",
        "energy_test_unbalanced",
        "bin_count",
    ]

    if return_detailed_output:
        # Calculate bin information
        df_stds = df.groupby("ws_bin")[std_cols].std()
        df_stds.columns = ["{}_std".format(c) for c in df_stds.columns]

        # Mean values of bins and power values
        df_means = df.groupby("ws_bin")[mean_cols].mean()
        df_means.columns = ["{}_mean".format(c) for c in df_means.columns]

        # Collect into a single dataframe
        df_per_ws_bin = pd.concat([df_means, df_stds, df_sums], axis=1)
        df_per_ws_bin["wd_bin"] = wd_bin[0]
        df_per_ws_bin["wd_bin_edges"] = df_freq["wd_bin_edges"]
        df_per_ws_bin["ws_bin_edges"] = df_freq["ws_bin_edges"]

        # Calculate unbalanced energy ratio for each wind speed bin
        df_per_ws_bin["energy_ratio_unbalanced"] = (
            df_per_ws_bin["energy_test_unbalanced"]
            / df_per_ws_bin["energy_ref_unbalanced"]
        )

        # Calculate (total) unbalanced energy ratio for all wind speeds
        energy_ratio_total_unbalanced = (
            df_per_ws_bin["energy_test_unbalanced"].sum()
            / df_per_ws_bin["energy_ref_unbalanced"].sum()
        )

        # Calculate total statistics
        total_means = df[mean_cols].mean()
        total_means = total_means.rename(
            dict(zip(mean_cols, ["{:s}_mean".format(c) for c in mean_cols]))
        )
        total_stds = df[std_cols].std()
        total_stds = total_stds.rename(
            dict(zip(std_cols, ["{:s}_std".format(c) for c in std_cols]))
        )

        # Get summation of energy and bin frequencies
        total_sums = df[["pow_ref", "pow_test", "freq"]].sum()
        total_sums = total_sums.rename(
            {
                "pow_ref": "energy_ref_unbalanced",
                "pow_test": "energy_test_unbalanced",
                "freq": "bin_count",
            }
        )

        df_per_wd_bin = pd.concat([total_means, total_stds, total_sums])
        df_per_wd_bin["wd_bin"] = wd_bin[0]
        df_per_wd_bin[
            "energy_ratio_unbalanced"
        ] = energy_ratio_total_unbalanced

    else:
        df_per_wd_bin = pd.DataFrame({"wd_bin": [wd_bin[0]]})
        df_per_ws_bin = pd.DataFrame(
            {
                "pow_ref_mean": (
                    df_sums["energy_ref_unbalanced"] / df_sums["bin_count"]
                ),
                "pow_test_mean": (
                    df_sums["energy_test_unbalanced"] / df_sums["bin_count"]
                ),
                "bin_count": df_sums["bin_count"],
            }
        )

    # Write bin frequencies to the dataframe and ensure normalization
    df_per_ws_bin["freq_balanced"] = df_freq["freq"] / df_freq["freq"].sum()
    df_per_ws_bin["freq_balanced"] = df_per_ws_bin["freq_balanced"].fillna(0)

    # Calculate normalized balanced energy for ref and test turbine
    df_per_ws_bin["energy_ref_balanced_norm"] = (
        df_per_ws_bin["pow_ref_mean"] * df_per_ws_bin["freq_balanced"]
    )
    df_per_ws_bin["energy_test_balanced_norm"] = (
        df_per_ws_bin["pow_test_mean"] * df_per_ws_bin["freq_balanced"]
    )

    # Compute total balanced energy ratio over all wind speeds
    df_per_wd_bin["energy_test_balanced_norm"] = (
        df_per_ws_bin["energy_test_balanced_norm"].sum()
    )
    df_per_wd_bin["energy_ref_balanced_norm"] = (
        df_per_ws_bin["energy_ref_balanced_norm"].sum()
    )

    energy_ratio_total_balanced = float(
        df_per_wd_bin["energy_test_balanced_norm"] /
        df_per_wd_bin["energy_ref_balanced_norm"]
    )

    if return_detailed_output:
        df_per_ws_bin["energy_ratio_balanced"] = (
            df_per_ws_bin["energy_test_balanced_norm"]
            / df_per_ws_bin["energy_ref_balanced_norm"]
        )
        df_per_wd_bin["energy_ratio_balanced"] = energy_ratio_total_balanced

        # Formatting
        df_per_wd_bin = pd.DataFrame(df_per_wd_bin).T
        df_per_wd_bin["bin_count"] = df_per_wd_bin["bin_count"].astype(int)

        df_per_wd_bin["wd_bin_edges"] = df_freq.iloc[0]["wd_bin_edges"]
        df_per_wd_bin = df_per_wd_bin.set_index("wd_bin")

        df_per_ws_bin["bin_count"] = df_per_ws_bin["bin_count"].astype(int)
        dict_out = {
            "df_per_wd_bin": df_per_wd_bin,
            "df_per_ws_bin": df_per_ws_bin,
        }
        return energy_ratio_total_balanced, dict_out

    return energy_ratio_total_balanced
