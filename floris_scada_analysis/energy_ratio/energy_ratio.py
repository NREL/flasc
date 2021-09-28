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

from floris_scada_analysis.dataframe_operations import dataframe_manipulations as dfm
from floris_scada_analysis.energy_ratio import energy_ratio_visualization as ervis
from floris_scada_analysis import utilities as fsut


class energy_ratio:
    """"""
    def __init__(
        self,
        df_in,
        wind_rose_function=None,
        verbose=False,
    ):
        self.verbose = verbose

        # Initialize dataframe
        self.set_df(df_in)

        # Initialize frequency functions
        self.wind_rose_function = wind_rose_function

    def set_df(self, df_in):
        if 'pow_ref' not in df_in.columns:
            raise KeyError('pow_ref column not found in dataframe. Cannot proceed.')
            # INFO: You can add such a column using dfm.set_pow_ref_by_*.

        # Copy minimal columns from df_in to df
        df = df_in[['wd', 'ws', 'pow_ref']].copy()

        self.df_full = df_in  # Save reference to full dataframe
        self.df = df  # Minimal dataframe

    def set_test_turbines(self, test_turbines):
        if not (type(test_turbines) is list):
            test_turbines = [test_turbines]
        self.test_turbines = test_turbines

        if "pow_test" in self.df.columns:
            self.df.drop(columns="pow_test")

        self.df["pow_test"] = dfm.get_column_mean(
            df=self.df_full,
            col_prefix="pow",
            turbine_list=self.test_turbines,
            circular_mean=False,
        )

    def _calculate_bins(self):
        # Define bin centers and bin widths
        ws_step = self.ws_step

        wd_step = self.wd_step
        wd_bin_width = self.wd_bin_width

        ws_labels = np.arange(
            np.min(self.df["ws"]) - 1.0e-6 + ws_step / 2.,
            np.max(self.df["ws"]) + 1.0e-6 + ws_step / 2.,
            ws_step
        )

        wd_labels = np.arange(
            np.min(self.df["wd"]) - 1.0e-6 + wd_step / 2.,
            np.max(self.df["wd"]) + 1.0e-6 + wd_step / 2.,
            wd_step
        )

        # Bin according to wind speed [never overlap between bins]
        for ws in ws_labels:
            ws_min = ws - ws_step / 2.
            ws_max = ws + ws_step / 2.
            ids = (self.df["ws"] > ws_min) & (self.df["ws"] <= ws_max)
            self.df.loc[ids, "ws_bin"] = ws

        # Bin according to wind direction [bin overlap if wd_bin_width > wd_step]
        df_list = [None for _ in range(len(wd_labels))]
        for ii, wd in enumerate(wd_labels):
            wd_min = wrap_360(wd - wd_bin_width / 2.)
            wd_max = wrap_360(wd + wd_bin_width / 2.)
            ids = (self.df["wd"] > wd_min) & (self.df["wd"] <= wd_max)
            df_subset = self.df.loc[ids].copy()
            df_subset["wd_bin"] = wd
            df_list[ii] = df_subset
        self.df = pd.concat(df_list, copy=False)

        # Make sure a float
        self.df["ws_bin"] = self.df["ws_bin"].astype(float)
        self.df["wd_bin"] = self.df["wd_bin"].astype(float)

        # Self the labels, these are all possible values
        self.ws_labels = ws_labels
        self.wd_labels = wd_labels

    def _get_df_freq(self):
        # Determine observed or annual freq
        if self.wind_rose_function is None:
            df_freq_observed = self.df[["ws_bin", "wd_bin"]].copy()
            df_freq_observed["freq"] = 1
            df_freq_observed = df_freq_observed.groupby(["ws_bin", "wd_bin"]).sum()
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

    def get_energy_ratio(
        self,
        test_turbines,
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=None,
        N=1,
        percentiles=[5., 95.]
    ):
        if self.df.shape[0] < 1:
            # Empty dataframe, do nothing
            self.energy_ratio_out = pd.DataFrame()
            self.energy_ratio_N = N
            return None

        if self.verbose:
            print('Calculating energy ratios with N = %d.' % N)

        # Set up a 'pow_test' column in the dataframe
        self.set_test_turbines(test_turbines)

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
            percentiles=percentiles
        )

        self.energy_ratio_out = energy_ratios
        self.energy_ratio_N = N

        return energy_ratios

    def plot_energy_ratio(self):
        # Simple linker function for easy usage
        ax = ervis.plot(self.energy_ratio_out)
        return ax


def _get_energy_ratios_all_wd_bins_bootstrapping(
    df_binned, df_freq=None, N=1, percentiles=[5., 95.]
    ):
    """Wrapper function that calculates the energy ratio for every wind
    direction bin in the dataframe. Thus, for every set of wind directions,
    the function '_get_energy_ratio_single_wd_bin_bootstrapping' is called.

    Args:
        df_binned ([type]): [description]
        df_freq ([type]): [description]
        N (int, optional): [description]. Defaults to 1.
        percentiles (list, optional): [description]. Defaults to [5., 95.].

    Returns:
        [type]: [description]
    """
    # Copy minimal dataframe
    df = df_binned[['ws_bin', 'wd_bin', 'pow_ref', 'pow_test']]

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
        result[wd_idx, :] = (
            _get_energy_ratio_single_wd_bin_bootstrapping(
                df_binned=df_subset,
                df_freq=df_freq_subset,
                N=N,
                percentiles=percentiles
            )
        )

    # Save energy ratios to the dataframe
    df_out = pd.DataFrame(
        result,
        columns=["baseline", "baseline_l", "baseline_u"]
    )

    # Save wind direction bins and bin count to dataframe
    df_out["wd_bin"] = unique_wd_bins
    _, df_out["N_bin"] = np.unique(df["wd_bin"], return_counts=True)

    return df_out


def _get_energy_ratio_single_wd_bin_bootstrapping(
    df_binned, df_freq=None, N=1, percentiles=[5., 95.]
    ):
    """Get the energy ratio for one particular wind direction bin and
    an array of wind speed bins. This function also includes bootstrapping
    functionality by increasing the number of bootstrap evaluations (N) to
    larger than 1. The bootstrap percentiles default to 5 % and 95 %.

    Args:
        df_binned ([type]): [description]
        df_freq ([type]): [description]
        N (int, optional): [description]. Defaults to 1.
        percentiles (list, optional): [description]. Defaults to [5., 95.].

    Returns:
        [type]: [description]
    """
    if df_freq is not None:
        # Renormalize frequencies
        df_freq = df_freq[["ws_bin", "freq"]].copy()
        df_freq["freq"] = df_freq["freq"] / df_freq["freq"].sum()

    # Get results excluding uncertainty
    results = _get_energy_ratio_single_wd_bin_nominal(
        df_binned=df_binned, df_freq=df_freq, randomize_df=False
    )

    # Add bootstrapping results, if necessary
    if N <= 1:
        results_array = np.array([results, results, results])
    else:
        # Get a bootstrap sample of range
        bootstrap_results = np.zeros([N, 1])
        for i in range(N):
            bootstrap_results[i, :] = (
                _get_energy_ratio_single_wd_bin_nominal(
                    df_binned=df_binned, df_freq=df_freq, randomize_df=True
                )
            )

        # Return the results in the order used in previous versions
        results_array = np.array([
            results,
            np.nanpercentile(bootstrap_results[:, 0], percentiles)[0],
            np.nanpercentile(bootstrap_results[:, 0], percentiles)[1],
        ])

    return results_array


def _get_energy_ratio_single_wd_bin_nominal(
    df_binned, df_freq=None, randomize_df=False
    ):
    # Copy over minimal dataframe
    df = df_binned[["ws_bin", "pow_ref", "pow_test"]].copy()

    # If resampling for boot-strapping
    if randomize_df:
        df = df.sample(frac=1, replace=True)

    if df_freq is None:
        ref_energy = np.nanmean(df_binned["pow_ref"])
        test_energy = np.nanmean(df_binned["pow_test"])

    else:
        df_freq = df_freq[["ws_bin", "freq"]].copy()

        # Group and sort by pow_ref and pow_test
        df_freq = df_freq.groupby(["ws_bin"]) .sum()
        df_grouped = df.groupby(["ws_bin"]).mean()
        df_ref = df_grouped["pow_ref"]
        df_test = df_grouped["pow_test"]

        # Ensure that dimensions match
        ws_bins = np.array(df_ref.index, dtype=float)
        ws_bins_freq = np.array(df_freq.index, dtype=float)
        if not np.array_equal(ws_bins, ws_bins_freq):
            ws_bins_array = np.concatenate([ws_bins, ws_bins_freq])
            ws_bins_array = np.unique(ws_bins_array)

            y = np.interp(ws_bins_array, ws_bins, df_ref)
            df_ref = pd.Series(data=y, index=ws_bins_array, name="pow_ref")

            y = np.interp(ws_bins_array, ws_bins, df_test)
            df_test = pd.Series(data=y, index=ws_bins_array, name="pow_test")

            # df_freq = df_freq.append({'ws_bin': i, 'observed_freq': np.nan, 'annual_freq': np.nan}, ignore_index=True)
            # df_freq = df_freq.sort(by='ws_bin')

        # Compute energy ratio
        ref_energy = np.dot(df_ref, df_freq["freq"])
        test_energy = np.dot(df_test, df_freq["freq"])
        # ref_energy = compute_expectation(df_ref, df_freq["freq"])
        # test_energy = compute_expectation(df_test, df_freq["freq"])

    energy_ratio = test_energy / ref_energy
    return energy_ratio


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
