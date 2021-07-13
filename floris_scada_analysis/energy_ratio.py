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
import os
import pandas as pd

from floris_scada_analysis import dataframe_manipulations as dfm
from floris_scada_analysis import energy_ratio_visualization as ervis
from floris_scada_analysis import utilities as fsut


def compute_expectation(fx, p_X):
        """
        Compute expected value of f(X), X ~ p_X(x).
        Inputs:
            fx - pandas Series / 1-D numpy - array of possible outcomes.
            p_X - pandas Series / 1-D numpy - distribution of X.
                                              May be supplied as 
                                              nonnormalized frequencies.
        Outputs:
            (anon) - float - Expected value of f(X)
        """
        p_X = p_X/p_X.sum() # Make sure distribution is valid

        return fx @ p_X


class energy_ratio:
    """"""

    def __init__(
        self,
        df_in,
        test_turbines,
        wd_step=2.0,
        ws_step=1.0,
        verbose=False,
    ):
        self.verbose = verbose
        self.df = pd.DataFrame()
        self._set_df(df_in)
        self.num_turbines = fsut.get_num_turbines(df_in)
        self.set_test_turbines(test_turbines, init=True)
        self.ws_step = ws_step
        self.wd_step = wd_step
        self.set_pow_test()

    def _set_df(self, df_in):
        if 'pow_ref' not in df_in.columns:
            raise KeyError('pow_ref column not found in dataframe. Cannot proceed.')
            # INFO: You can add such a column using dfm.set_pow_ref_by_*.

        # Copy minimal columns from df_in to df
        df = df_in[['wd', 'ws', 'pow_ref']].copy()

        self.df_full = df_in  # Save reference to full dataframe
        self.df = df  # Minimal dataframe

    def set_test_turbines(self, test_turbines, init=False):
        if not (type(test_turbines) is list):
            test_turbines = [test_turbines]
        self.test_turbines = test_turbines
        if not init:
            self.set_pow_test()

    def set_pow_test(self):
        self.df["pow_test"] = dfm.get_column_mean(
            df=self.df_full,
            col_prefix="pow",
            turbine_list=self.test_turbines,
            circular_mean=False,
        )

    def setup_directions_and_frequencies(self):
        # Define some local values
        ws_step = self.ws_step
        wd_step = self.wd_step
        wd_min = np.min(self.df["wd"]) - 1.0e-6
        wd_max = np.max(self.df["wd"]) + 1.0e-6
        ws_min = np.min(self.df["ws"]) - 1.0e-6
        ws_max = np.max(self.df["ws"]) + 1.0e-6

        ws_edges = np.arange(ws_min, ws_max + ws_step, ws_step)
        ws_labels = ws_edges[:-1] + ws_step / 2.0  # Bin means
        wd_edges = np.arange(wd_min, wd_max + wd_step, wd_step)
        wd_labels = wd_edges[:-1] + wd_step / 2.0  # Bin means

        # Notice both bins labeled by the middle
        self.df["ws_bin"] = pd.cut(self.df["ws"], ws_edges, labels=ws_labels)
        self.df["wd_bin"] = pd.cut(self.df["wd"], wd_edges, labels=wd_labels)

        # Make sure a float
        self.df["ws_bin"] = self.df["ws_bin"].astype(float)
        self.df["wd_bin"] = self.df["wd_bin"].astype(float)

        # Self the labels, these are all possible values
        self.ws_labels = ws_labels
        self.wd_labels = wd_labels

        # Save the edges too
        self.ws_edges = ws_edges
        self.wd_edges = wd_edges

        # Build up the observed and annual frequency tables
        self._get_df_freq()

    def _build_observed_freq(self):
        df_freq_observed = self.df[["ws_bin", "wd_bin"]].copy()
        df_freq_observed["freq"] = 1
        df_freq_observed = df_freq_observed.groupby(["ws_bin", "wd_bin"]).sum()

        indices = list(product(self.ws_labels, self.wd_labels))
        self.df_freq_observed = (
            df_freq_observed.reindex(indices).fillna(0).reset_index()
        )

    def _build_annual_freq(self, filename="wind_rose_annual_function.p"):
        if os.path.exists(filename):
            # Load the saved annual frequency function
            annual_interp_function = pd.read_pickle("wind_rose_annual_function.p")

            num_bins = len(self.ws_labels) * len(self.wd_labels)
            ws_array = np.zeros(num_bins)
            wd_array = np.zeros(num_bins)
            freq_array = np.zeros(num_bins)

            for idx, (ws, wd) in enumerate(product(self.ws_labels, self.wd_labels)):
                ws_array[idx] = ws
                wd_array[idx] = wd
                freq_array[idx] = annual_interp_function(ws, wd)
            self.df_freq_annual = pd.DataFrame(
                {"ws_bin": ws_array, "wd_bin": wd_array, "freq": freq_array}
            )
        else:
            self.df_freq_annual = self.df_freq_observed

    def _get_df_freq(self):
        # Determine observed and annual freq
        self._build_observed_freq()
        self._build_annual_freq()

        # Merge observed and annual distributions
        df_freq = self.df_freq_observed.copy()
        df_freq.columns = ['ws_bin', 'wd_bin', 'observed_freq']
        df_freq = df_freq.merge(self.df_freq_annual,how='left', on=['ws_bin', 'wd_bin'])
        df_freq.columns = ['ws_bin', 'wd_bin', 'observed_freq', 'annual_freq']

        self.df_freq = df_freq

    def get_energy_ratio(self, N=1):
        if self.df.shape[0] < 1:
            result = pd.DataFrame()
            return None

        if self.verbose:
            print('Calculating energy ratio with N = %d.' % N)
        if 'ws_bin' not in self.df.columns:
            if self.verbose:
                print('  df has not yet been binned. Binning data first...')
            self.setup_directions_and_frequencies()
        elif self.verbose:
            print('  df has already been binned. Loading existing bins.')

        # Define the energy frame class
        energy_frame = energy_ratio_for_wd_array(df=df, df_freq=self.df_freq)

        # Retrieve the right function
        result = energy_frame.get_energy_ratios(N=N)
        self.energy_ratio_out = result
        self.energy_ratio_N = N

        return result

    def plot_energy_ratio(self):
        # Simple linker function for easy usage
        ax = ervis.plot_single_curve(self.energy_ratio_out)
        return ax


# Legacy class from Paul Fleming
class energy_ratio_for_wd_array:
    """
    Top level analysis class for storing the data used across the analysis

    Args:

    Returns:
        TBD
    """

    def __init__(self, df, df_freq):

        self.df = df[['ws_bin', 'wd_bin', 'pow_ref', 'pow_test']]

        # Renormalize frequecies
        df_freq["observed_freq"] = df_freq.observed_freq / df_freq.observed_freq.sum()
        df_freq["annual_freq"] = df_freq.annual_freq / df_freq.annual_freq.sum()
        self.df_freq = df_freq

        # Save some relevent info
        self.wd_bin = np.array(sorted(df.wd_bin.unique()))
        self.ws_bin = np.array(sorted(df.ws_bin.unique()))

    def get_energy_ratios(self, use_observed_freq=True, N=1):
        result = np.zeros([len(self.wd_bin), 3])
        for wd_idx, wd in enumerate(self.wd_bin):
            df_sub = (
                self.df[self.df.wd_bin == wd]
                # .reset_index(drop=True)
                # .drop(["wd_bin"], axis="columns")
            )
            df_freq = (
                self.df_freq[self.df_freq.wd_bin == wd]
                # .reset_index(drop=True)
                # .drop(["wd_bin"], axis="columns")
            )
            ec = energy_ratio_for_single_wd_bin(df_sub, df_freq, self.ws_bin)
            result[wd_idx, :] = ec.get_energy_ratio(
                use_observed_freq=use_observed_freq, N=N
            )

        df_res = pd.DataFrame(result, columns=["baseline", "baseline_l", "baseline_u"])

        df_res["wd_bin"] = self.wd_bin
        _, df_res["N_bin"] = np.unique(self.df.wd_bin, return_counts=True)

        return df_res


# Legacy class from Paul Fleming
class energy_ratio_for_single_wd_bin:
    def __init__(self, df, df_freq, ws_bin):

        # Copy minimal dataframe
        self.df = df[["ws_bin", "pow_ref", "pow_test"]]

        # Renormalize frequecies
        df_freq["observed_freq"] = df_freq.observed_freq / df_freq.observed_freq.sum()
        df_freq["annual_freq"] = df_freq.annual_freq / df_freq.annual_freq.sum()
        self.df_freq = df_freq

        # Save the bins
        self.ws_bin = ws_bin

    def _get_energy_ratio_nominal(self, use_observed_freq=True, randomize_df=False):

        # Local copies
        df = self.df #.copy()
        df_freq = self.df_freq #.copy()

        # If resampling for boot-strapping
        if randomize_df:
            df = df.sample(frac=1, replace=True)

        df_group = (
            df[["ws_bin", "pow_ref", "pow_test"]]
            .groupby(["ws_bin"])
            .mean()
        )

        df_ref = df_group["pow_ref"]
        df_test = df_group["pow_test"]

        df_freq = (
            df_freq[["ws_bin", "observed_freq", "annual_freq"]]
            .groupby(["ws_bin"])
            .sum()
        )

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

        if use_observed_freq:
            freq_signal = "observed_freq"
        else:
            freq_signal = "annual_freq"

        ref_energy = compute_expectation(df_ref, df_freq[freq_signal])
        test_energy = compute_expectation(df_test, df_freq[freq_signal])
        energy_ratio = test_energy / ref_energy

        return energy_ratio

    def get_energy_ratio(self, use_observed_freq=True, N=1, percentiles=[10, 90]):

        # Get central results
        results = self._get_energy_ratio_nominal(
            use_observed_freq=use_observed_freq, randomize_df=False
        )

        # Add bootstrapping results, if necessary
        if N <= 1:
            results_array = np.array([results, results, results])
        else:
            # Get a bootstrap sample of range
            bootstrap_results = np.zeros([N, 1])
            for i in range(N):
                bootstrap_results[i, :] = self._get_energy_ratio_nominal(
                    use_observed_freq=use_observed_freq,
                    randomize_df=True
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
