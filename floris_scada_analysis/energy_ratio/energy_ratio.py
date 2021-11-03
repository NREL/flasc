# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from itertools import product
import numpy as np
import pandas as pd

from floris.utilities import wrap_360
from pandas.core.base import DataError

from floris_scada_analysis.dataframe_operations import (
    dataframe_manipulations as dfm,
)
from floris_scada_analysis.energy_ratio import (
    energy_ratio_visualization as ervis,
)


class energy_ratio:
    """This class is used to calculate the energy ratios for a single
    dataframe with measurements, either from FLORIS or from SCADA data.
    This class supports bootstrapping for uncertainty quantification,
    automatic derivation of the frequency of bins based on occurrence
    in the provided dataset, and various choices for binning and daa
    discretization.
    """

    def __init__(self, df_in, wind_rose_function=None, verbose=False):
        """Initialization of the class.

        Args:
            df_in ([pd.DataFrame]): The dataframe provided by the user. This
            dataframe should have the following columns:
                * Reference wind direction for the test turbine, 'wd'
                * Reference wind speed for the test turbine, 'ws'
                * Power production of every turbine: pow_000, pow_001, ...
                * Reference power production used to normalize the energy
                    ratio: 'pow_ref'
            wind_rose_function ([pd.DataFrame], optional): This defines the
            occurrence of each wind direction and wind speed bin. If None is
            specified, the occurrence of each bin is derived from the provided
            data, df_in. Defaults to None.
            verbose (bool, optional): Print to console. Defaults to False.
        """
        self.verbose = verbose

        # Initialize dataframe
        self._set_df(df_in)

        # Initialize frequency functions
        self.wind_rose_function = wind_rose_function

    # Private methods

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
            #   from floris_scada_analysis.dataframe_operations import \
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

        self.df = self.df_full[["wd", "ws", "pow_ref"]].copy()
        self.df["pow_test"] = dfm.get_column_mean(
            df=self.df_full,
            col_prefix="pow",
            turbine_list=self.test_turbines,
            circular_mean=False,
        )

    def _calculate_bins(self):
        """This function bins the data in the minimal dataframe, self.df,
        into the respective wind direction and wind speed bins. Note that
        there might be bin overlap if the specified wd_bin_width is larger
        than the bin step size. This code will copy dataframe rows that fall
        into multiple bins, effectively increasing the sample size.
        """
        # Define bin centers and bin widths
        ws_step = self.ws_step
        wd_step = self.wd_step
        wd_bin_width = self.wd_bin_width

        ws_labels = np.arange(
            np.min(self.df["ws"]) - 1.0e-6 + ws_step / 2.0,
            np.max(self.df["ws"]) + 1.0e-6 + ws_step / 2.0,
            ws_step,
        )

        wd_labels = np.arange(
            np.min(self.df["wd"]) - 1.0e-6 + wd_step / 2.0,
            np.max(self.df["wd"]) + 1.0e-6 + wd_step / 2.0,
            wd_step,
        )

        # Bin according to wind speed. Note that data never falls into
        # multiple wind speed bins at the same time.
        for ws in ws_labels:
            ws_min = float(ws - ws_step / 2.0)
            ws_max = float(ws + ws_step / 2.0)
            ws_interval = pd.Interval(ws_min, ws_max, "right")
            ids = (self.df["ws"] > ws_min) & (self.df["ws"] <= ws_max)
            self.df.loc[ids, "ws_bin"] = ws
            self.df.loc[ids, "ws_bin_edges"] = ws_interval

        # Bin according to wind direction. Note that data can fall into
        # multiple wind direction bins at the same time, if wd_bin_width is
        # larger than the wind direction binning step size, wd_step. If so,
        # data will be copied and the sample size is effectively increased
        # so that every relevant bin has that particular measurement.
        df_list = [None for _ in range(len(wd_labels))]
        for ii, wd in enumerate(wd_labels):
            wd_min = float(wrap_360(wd - wd_bin_width / 2.0))
            wd_max = float(wrap_360(wd + wd_bin_width / 2.0))
            wd_interval = pd.Interval(wd_min, wd_max, "right")
            ids = (self.df["wd"] > wd_min) & (self.df["wd"] <= wd_max)
            df_subset = self.df.loc[ids].copy()
            df_subset["wd_bin"] = wd
            df_subset["wd_bin_edges"] = wd_interval
            df_list[ii] = df_subset
        self.df = pd.concat(df_list, copy=False)

        # Make sure a float
        self.df["ws_bin"] = self.df["ws_bin"].astype(float)
        self.df["wd_bin"] = self.df["wd_bin"].astype(float)

        # Self the labels, these are all possible values
        self.ws_labels = ws_labels
        self.wd_labels = wd_labels

    def _get_df_freq(self):
        """This function derives the frequency of occurrence of each bin
        (wind direction and wind speed) from the binned dataframe. The
        found values are used in the energy ratio equation to weigh the
        power productions of each bin according to their frequency of
        occurrence.
        """
        # Determine observed or annual freq
        if self.wind_rose_function is None:
            df_freq_observed = self.df[["ws_bin", "wd_bin"]].copy()
            df_freq_observed["freq"] = 1
            df_freq_observed = df_freq_observed.groupby(
                ["ws_bin", "wd_bin"]
            ).sum()
            df_freq_observed["freq"] = df_freq_observed["freq"].astype(int)

            indices = list(product(self.ws_labels, self.wd_labels))
            df_freq_observed = (
                df_freq_observed.reindex(indices).fillna(0).reset_index()
            )
            self.df_freq = df_freq_observed

        else:
            self.df_freq = pd.DataFrame()  # ...
            raise NotImplementedError(
                "This functionality is not yet implemented."
            )
            # num_bins = len(self.ws_labels) * len(self.wd_labels)
            # ws_array = np.zeros(num_bins)
            # wd_array = np.zeros(num_bins)
            # freq_array = np.zeros(num_bins)

            # for idx, (ws, wd) in enumerate(product(self.ws_labels, self.wd_labels)):
            #     ws_array[idx] = ws
            #     wd_array[idx] = wd
            #     freq_array[idx] = annual_interp_function(ws, wd)
            # self.df_freq_annual = pd.DataFrame(
            #     {"ws_bin": ws_array, "wd_bin": wd_array, "freq": freq_array}
            # )

    # Public methods

    def get_energy_ratio(
        self,
        test_turbines,
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=None,
        N=1,
        percentiles=[5.0, 95.0],
        return_detailed_output=False,
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
            N (int, optional): Number of bootstrap evaluations for
                uncertainty quantification (UQ). If N=1, will not perform
                any uncertainty quantification. Defaults to 1.
            percentiles (list, optional): Confidence bounds for the
                uncertainty quantification in percents. This value is only
                relevant if N > 1 is specified. Defaults to [5., 95.].

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

        # Set up binning
        self.ws_step = ws_step
        self.wd_step = wd_step
        if wd_bin_width is None:
            wd_bin_width = wd_step
        self.wd_bin_width = wd_bin_width
        self._calculate_bins()

        # Get probability distribution of bins
        if N > 1:
            self._get_df_freq()
        else:
            self.df_freq = None

        # Calculate the energy ratio for all bins
        energy_ratios = _get_energy_ratios_all_wd_bins_bootstrapping(
            df_binned=self.df,
            df_freq=self.df_freq,
            N=N,
            percentiles=percentiles,
            return_detailed_output=return_detailed_output,
        )

        self.energy_ratio_out = energy_ratios
        self.energy_ratio_N = N

        return energy_ratios

    def plot_energy_ratio(self):
        """This function plots the energy ratio against the wind direction,
        potentially with uncertainty bounds if N > 1 was specified by
        the user. One must first run get_energy_ratio() before attempting
        to plot the energy ratios.

        Returns:
            ax [plt.Axes]: Axis handle for the figure.
        """
        return ervis.plot(self.energy_ratio_out)


# Support functions not included in energy_ratio class


def _get_energy_ratios_all_wd_bins_bootstrapping(
    df_binned,
    df_freq=None,
    N=1,
    percentiles=[5.0, 95.0],
    return_detailed_output=False,
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
        min_cols = ["wd", "ws", "ti", "ws_bin", "wd_bin", "pow_ref", "pow_test"]
    else:
        min_cols = ["wd", "ws", "ws_bin", "wd_bin", "pow_ref", "pow_test"]
    df = df_binned[min_cols]

    # Save some relevant info
    unique_wd_bins = np.unique(df.wd_bin)
    # unique_ws_bins = np.unique(df.ws_bin)

    # Now calculate the actual energy ratios
    result = np.zeros([len(unique_wd_bins), 3])
    for wd_idx, wd in enumerate(unique_wd_bins):
        df_subset = df[df["wd_bin"] == wd]
        if df_freq is None:
            df_freq_subset = None
        else:
            df_freq_subset = df_freq[df_freq["wd_bin"] == wd]
        result[wd_idx, :] = _get_energy_ratio_single_wd_bin_bootstrapping(
            df_binned=df_subset,
            df_freq=df_freq_subset,
            N=N,
            percentiles=percentiles,
            return_detailed_output=return_detailed_output,
        )

    # Save energy ratios to the dataframe
    df_out = pd.DataFrame(
        result, columns=["baseline", "baseline_l", "baseline_u"]
    )

    # Save wind direction bins and bin count to dataframe
    df_out["wd_bin"] = unique_wd_bins
    _, df_out["N_bin"] = np.unique(df["wd_bin"], return_counts=True)

    return df_out


def _get_energy_ratio_single_wd_bin_bootstrapping(
    df_binned,
    df_freq=None,
    N=1,
    percentiles=[5.0, 95.0],
    return_detailed_output=False,
):
    """Get the energy ratio for one particular wind direction bin and
    an array of wind speed bins. This function also includes bootstrapping
    functionality by increasing the number of bootstrap evaluations (N) to
    larger than 1. The bootstrap percentiles default to 5 % and 95 %.
    """
    if df_freq is not None:
        # Renormalize frequencies
        df_freq = df_freq[["ws_bin", "freq"]].copy()
        df_freq["freq"] = df_freq["freq"] / df_freq["freq"].sum()

    # Get results excluding uncertainty
    results = _get_energy_ratio_single_wd_bin_nominal(
        df_binned=df_binned, df_freq=df_freq, randomize_df=False,
        return_detailed_output=return_detailed_output
    )

    # Add bootstrapping results, if necessary
    if N <= 1:
        results_array = np.array([results, results, results])
    else:
        # Get a bootstrap sample of range
        bootstrap_results = np.zeros([N, 1])
        for i in range(N):
            bootstrap_results[i, :] = _get_energy_ratio_single_wd_bin_nominal(
                df_binned=df_binned,
                df_freq=df_freq,
                randomize_df=True,
                return_detailed_output=False
            )

        # Return the results in the order used in previous versions
        results_array = np.array(
            [
                results,
                np.nanpercentile(bootstrap_results[:, 0], percentiles)[0],
                np.nanpercentile(bootstrap_results[:, 0], percentiles)[1],
            ]
        )

    return results_array


def _get_energy_ratio_single_wd_bin_nominal(
    df_binned, df_freq=None, randomize_df=False, return_detailed_output=False
):
    """Get the energy ratio for one particular wind direction bin and
    an array of wind speed bins. This function performs a single
    calculation of the energy ratios without uncertainty quantification.
    """
    # Copy minimal dataframe
    if "ti" in df_binned.columns:
        min_cols = ["wd", "ws", "ti", "ws_bin", "pow_ref", "pow_test"]
        mean_cols = ["wd", "ws", "ti", "pow_ref", "pow_test"]
        std_cols = ["wd", "ws", "ti", "pow_ref", "pow_test"]
    else:
        min_cols = ["wd", "ws", "ws_bin", "pow_ref", "pow_test"]
        mean_cols = ["wd", "ws", "pow_ref", "pow_test"]
        std_cols = ["wd", "ws", "pow_ref", "pow_test"]
    df = df_binned[min_cols]

    # Check for faulty measurements
    if np.any(np.isnan(df)):
        raise DataError("All entries in dataframe must be non-NaN.")

    # If resampling for boot-strapping, randomize dataframe
    if randomize_df:
        df = df.sample(frac=1, replace=True)

    # Reference and test turbine energy
    df["freq"] = 1
    df_sums = df.groupby("ws_bin")[["pow_ref", "pow_test", "freq"]].sum()
    df_sums.columns = [
        "energy_ref_unbalanced",
        "energy_test_unbalanced",
        "bin_count"
    ]

    # Calculate bin information
    df_stds = df.groupby("ws_bin")[std_cols].std()
    df_stds.columns = ["{}_std".format(c) for c in df_stds.columns]

    # Mean values of bins and power values
    df_means = df.groupby("ws_bin")[mean_cols].mean()
    df_means.columns = ["{}_mean".format(c) for c in df_means.columns]

    # Collect into a single dataframe
    df_info = pd.concat([df_means, df_stds, df_sums], axis=1)

    # Calculate unbalanced energy ratio for each wind speed bin
    df_info["energy_ratio_unbalanced"] = (
        df_info["energy_test_unbalanced"] /
        df_info["energy_ref_unbalanced"]
    )

    # Calculate (total) unbalanced energy ratio for all wind speeds
    energy_ratio_total_unbalanced = (
        df_info["energy_test_unbalanced"].sum() /
        df_info["energy_ref_unbalanced"].sum()
    )

    # Calculate total statistics
    total_means = df[mean_cols].mean()
    total_means = total_means.rename(
        {
            "wd": "wd_mean",
            "ws": "ws_mean",
            "pow_ref": "pow_ref_mean",
            "pow_test": "pow_test_mean",
        }
    )
    total_stds = df[std_cols].std()
    total_stds = total_stds.rename(
        {
            "wd": "wd_std",
            "ws": "ws_std",
            "pow_ref": "pow_ref_std",
            "pow_test": "pow_test_std",
        }
    )
    total_sums = df[["pow_ref", "pow_test", "freq"]].sum()
    total_sums = total_sums.rename(
        {
            "pow_ref": "energy_ref_unbalanced",
            "pow_test": "energy_test_unbalanced",
            "freq": "bin_count"
        }
    )
    df_total = pd.concat([total_means, total_stds, total_sums])
    df_total["energy_ratio_unbalanced"] = energy_ratio_total_unbalanced

    if df_freq is None:
        # Balanced energy ratio is equal to unbalanced energy ratio
        df_info["freq_balanced"] = df_info["bin_count"] / df_info["bin_count"].sum()
    else:
        # Derive
        df_info["freq_balanced"] = df_freq["freq"]
        if (np.abs(df_info["freq_balanced"].sum() - 1.0) > 0.00001):
            raise DataError("Provided bin frequencies do not add up to 1.0.")

    df_info["energy_ref_balanced_norm"] = df_info["pow_ref_mean"] * df_info["freq_balanced"]
    df_info["energy_test_balanced_norm"] = df_info["pow_test_mean"] * df_info["freq_balanced"]
    df_info["energy_ratio_balanced"] = (
        df_info["energy_test_balanced_norm"] /
        df_info["energy_ref_balanced_norm"]
    )

    # Compute energy ratio
    energy_ratio_total_balanced = (
        df_info["energy_test_balanced_norm"].sum() /
        df_info["energy_ref_balanced_norm"].sum()
    )
    df_total["energy_ratio_balanced"] = energy_ratio_total_balanced

    # Formatting
    df_total = pd.DataFrame(df_total).T
    df_total["bin_count"] = df_total["bin_count"].astype(int)
    df_info["bin_count"] = df_info["bin_count"].astype(int)

    return energy_ratio_total_balanced


# def compute_expectation(fx, p_X):
#         """
#         Compute expected value of f(X), X ~ p_X(x).
#         Inputs:
#             fx - pandas Series / 1-D numpy - array of possible outcomes.
#             p_X - pandas Series / 1-D numpy - distribution of X.
#                                               May be supplied as
#                                               nonnormalized frequencies.
#         Outputs:
#             (anon) - float - Expected value of f(X)
#         """
#         p_X = p_X/p_X.sum()  # Make sure distribution is valid

#         return fx @ p_X
