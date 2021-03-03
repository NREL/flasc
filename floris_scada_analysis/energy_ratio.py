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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from floris_scada_analysis import dataframe_manipulations as dfm


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
        df,
        test_turbines,
        ref_turbines,
        dep_turbines=[],
        wd_step=2.0,
        ws_step=1.0,
        verbose=False,
    ):
        self.verbose = verbose

        self._set_df(df)
        self.num_turbines = dfm.get_num_turbines(self.df)

        self.set_test_turbines(test_turbines, init=True)
        self.set_ref_turbines(ref_turbines, init=True)
        self.set_dep_turbines(dep_turbines)

        self.ws_step = ws_step
        self.wd_step = wd_step

        self.filter_df_by_status()
        self.set_ref_test_power()


    def _set_df(self, df):
        if 'category' not in df.columns:
            if self.verbose:
                print("Did not find 'category' column. Adding one with all values 'baseline'.")
            df['category'] = 'baseline'
        if 'status' not in df.columns:
            if self.verbose:
                print("Did not find 'status_' columns in df. Will assume all measurements are correct.")

        self.categories = list(np.unique(df['category']))
        if len(self.categories) > 2:
            raise KeyError('Cannot have more than 2 categories in your dataframe.')
        else:
            if self.verbose:
                print('Your dataframe has categories', self.categories)
        self.df = df

    def _set_turbines_all(self):
        self.turbines_all = np.unique(
            [*self.test_turbines, *self.ref_turbines, *self.dep_turbines]
        )

    def set_dep_turbines(self, dep_turbines):
        if not (type(dep_turbines) is list):
            dep_turbines = [dep_turbines]
        self.dep_turbines = dep_turbines
        self._set_turbines_all()

    def set_ref_turbines(self, ref_turbines, init=False):
        if not (type(ref_turbines) is list):
            ref_turbines = [ref_turbines]
        self.ref_turbines = ref_turbines
        if not init:
            self._set_turbines_all()
            self.set_ref_test_power()

    def set_test_turbines(self, test_turbines, init=False):
        if not (type(test_turbines) is list):
            test_turbines = [test_turbines]
        self.test_turbines = test_turbines
        if not init:
            self._set_turbines_all()
            self.set_ref_test_power()

    def set_ref_test_power(self):
        self.df["ref_power"] = dfm.get_column_mean(
            df=self.df,
            col_prefix="pow",
            turbine_list=self.ref_turbines,
            circular_mean=False,
        )
        self.df["test_power"] = dfm.get_column_mean(
            df=self.df,
            col_prefix="pow",
            turbine_list=self.test_turbines,
            circular_mean=False,
        )

    def filter_df_by_status(self):
        status_cols = ["status_%03d" % ti for ti in self.turbines_all]
        for c in status_cols:
            if not (c in self.df.columns):
                if self.verbose:
                    print(
                        "Column "
                        + c
                        + " does not exist or was already"
                        + " filtered. Assuming all values are 1 (good). "
                    )
                self.df[c] = int(1)

        all_status_okay = self.df[status_cols].sum(axis=1) == len(self.turbines_all)
        self.df = self.df[all_status_okay]
        self.df = self.df.drop(columns=status_cols)

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

        # Extract the interesting parts for the energy ratio frame
        rlvnt_cols = ['ws_bin', 'wd_bin', 'ref_power', 'test_power', 'category']

        # Define the energy frame class
        energy_frame = Energy_Frame(df=self.df[rlvnt_cols],
                                    df_freq=self.df_freq,
                                    category=self.categories)

        # Retrieve the right function
        if len(self.categories) == 1:
            result = energy_frame.get_1_cat_energy_ratio_array_with_range(N=N)
        else:
            result = energy_frame.get_2_cat_energy_ratio_array_with_range(N=N)

        self.energy_ratio_out = result
        self.energy_ratio_N = N

        return result

    def plot_energy_ratio(self):
        fig, ax = plt.subplots()
        result = self.energy_ratio_out
        ax.plot(result.wd_bin, result.baseline)
        ax.fill_between(result.wd_bin, result.baseline_l, result.baseline_u, alpha=0.15)
        if len(self.categories) > 1:
            ax.plot(result.wd_bin, result.controlled)
            ax.fill_between(result.wd_bin, result.controlled_l, result.controlled_u, alpha=0.15)
        plt.legend(self.categories)
        plt.title(str(['Test turbines:', self.test_turbines,
                  'Ref. turbines:', self.ref_turbines,
                  'Dep. turbines:', self.dep_turbines])[1:-1])
        plt.xlabel('Wind direction (degrees)')
        plt.ylabel('Energy ratio (-)')
        plt.grid(True)
        return fig, ax


# Legacy class from Paul
class Energy_Frame:
    """
    Top level analysis class for storing the data used across the analysis

    Args:

    Returns:
        TBD
    """

    def __init__(self, df, df_freq, category):

        self.df = df

        # Renormalize frequecies
        df_freq["observed_freq"] = df_freq.observed_freq / df_freq.observed_freq.sum()
        df_freq["annual_freq"] = df_freq.annual_freq / df_freq.annual_freq.sum()
        self.df_freq = df_freq

        # Save some relevent info
        self.wd_bin = np.array(sorted(df.wd_bin.unique()))
        self.ws_bin = np.array(sorted(df.ws_bin.unique()))
        # self.wd_bin = np.array(sorted(df_freq.wd_bin.unique()))
        # self.ws_bin = np.array(sorted(df_freq.ws_bin.unique()))

        self.category = category

    def get_column(self, wd):
        df_sub = (
            self.df[self.df.wd_bin == wd]
            .reset_index(drop=True)
            .drop(["wd_bin"], axis="columns")
        )
        df_freq = (
            self.df_freq[self.df_freq.wd_bin == wd]
            .reset_index(drop=True)
            .drop(["wd_bin"], axis="columns")
        )
        return Energy_Column(df_sub, df_freq, self.ws_bin, self.category)

    def get_1_cat_energy_ratio_array_with_range(self, use_observed_freq=True, N=100):

        result = np.zeros([len(self.wd_bin), 3])

        for wd_idx, wd in enumerate(self.wd_bin):
            # print(wd)
            ec = self.get_column(wd)
            result[wd_idx, :] = ec.get_1cat_energy_ratio_with_range(
                use_observed_freq=use_observed_freq, N=N
            )

        df_res = pd.DataFrame(result, columns=["baseline", "baseline_l", "baseline_u"])
        df_res["wd_bin"] = self.wd_bin

        return df_res

    def get_2_cat_energy_ratio_array_with_range(self, use_observed_freq=True, N=100):

        result = np.zeros([len(self.wd_bin), 12])

        for wd_idx, wd in enumerate(self.wd_bin):
            # print(wd)
            ec = self.get_column(wd)
            result[wd_idx, :] = ec.get_2cat_energy_ratio_with_range(
                use_observed_freq=use_observed_freq, N=N
            )

        df_res = pd.DataFrame(
            result,
            columns=[
                "baseline",
                "baseline_l",
                "baseline_u",
                "controlled",
                "controlled_l",
                "controlled_u",
                "diff",
                "diff_l",
                "diff_u",
                "per",
                "per_l",
                "per_u",
            ],
        )
        df_res["wd_bin"] = self.wd_bin

        return df_res


# Legacy class from Paul
class Energy_Column:
    def __init__(self, df, df_freq, ws_bin, category):

        self.df = df

        # Renormalize frequecies
        df_freq["observed_freq"] = df_freq.observed_freq / df_freq.observed_freq.sum()
        df_freq["annual_freq"] = df_freq.annual_freq / df_freq.annual_freq.sum()
        self.df_freq = df_freq

        # Save the bins
        self.ws_bin = ws_bin
        self.category = category

    def get_energy_ratio(self, use_observed_freq=True, randomize_df=False):

        # Local copies
        df = self.df.copy()
        df_freq = self.df_freq.copy()
        ws_bin = self.ws_bin
        category = self.category

        # If resampling for boot-strapping
        if randomize_df:
            df = df.sample(frac=1, replace=True)

        # Remove bins with unmatched categories
        for ws in ws_bin:
            for cg_idx, cg in enumerate(category):
                if not (((df.ws_bin == ws) & (df.category == cg)).any()):
                    # print('Cat: %s is missing %d m/s, removing' % (cg, ws))
                    df = df[df.ws_bin != ws]
                    df_freq = df_freq[df_freq.ws_bin != ws]

        # Check for empty frame
        if df.shape[0] == 0:
            return np.zeros(len(category)) * np.nan

        df_group = (
            df[["ws_bin", "category", "ref_power", "test_power"]]
            .groupby(["ws_bin", "category"])
            .mean()
        )
        df_ref = df_group[["ref_power"]].unstack()
        df_ref.columns = [c[1] for c in df_ref.columns]
        df_test = df_group[["test_power"]].unstack()
        df_test.columns = [c[1] for c in df_test.columns]

        df_freq = (
            df_freq[["ws_bin", "observed_freq", "annual_freq"]]
            .groupby(["ws_bin"])
            .sum()
        )

        # # Ensure that dimensions match
        # ws_bins_ref = np.array(df_ref.reset_index()['ws_bin'])
        # ws_bins_test = np.array(df_ref.reset_index()['ws_bin'])
        # ws_bins_freq = np.array(df_freq.reset_index()['ws_bin'])
        # if (not all(ws_bins_ref==ws_bins_test) or
        #     not all(ws_bins_ref==ws_bins_freq)):
        #     ws_bins_array = np.concatenate([ws_bins_ref, ws_bins_test, ws_bins_freq])
        #     ws_bins_array = np.unique(ws_bins_array)
        #     for i in ws_bins_array:
        #         if i not in list(df_ref.reset_index()['ws_bin']):
        #             df_ref = df_ref.reset_index()
        #             df_ref = df_ref.append({'ws_bin': i, 'baseline': np.nan}, ignore_index=True)
        #             df_ref = df_ref.set_index('ws_bin').sort_index()
        #         if i not in list(df_test.reset_index()['ws_bin']):
        #             df_test = df_test.reset_index()
        #             df_test = df_test.append({'ws_bin': i, 'baseline': np.nan}, ignore_index=True)
        #             df_test = df_test.set_index('ws_bin').sort_index()
        #         if i not in list(df_test.reset_index()['ws_bin']):
        #             df_freq = df_freq.append({'ws_bin': i, 'observed_freq': np.nan, 'annual_freq': np.nan}, ignore_index=True)
        #             df_freq = df_freq.sort(by='ws_bin')

        results = np.zeros(len(category))
        if use_observed_freq:
            freq_signal = "observed_freq"
        else:
            freq_signal = "annual_freq"
        for catg_idx, catg in enumerate(category):
            ref_energy = compute_expectation(df_ref[catg], df_freq[freq_signal])
            test_energy = compute_expectation(df_test[catg], df_freq[freq_signal])
            results[catg_idx] = test_energy / ref_energy

        return results

    def get_1cat_energy_ratio(self, use_observed_freq=True, randomize_df=False):

        results = self.get_energy_ratio(
            use_observed_freq=use_observed_freq, randomize_df=randomize_df
        )

        # Return with the difference as a convience
        return np.array([results[0]])

    def get_1cat_energy_ratio_with_range(
        self, use_observed_freq=True, N=100, percentiles=[10, 90]
    ):

        # Get central results
        results = self.get_1cat_energy_ratio(
            use_observed_freq=use_observed_freq, randomize_df=False
        )

        # Get a bootstrap sample of range
        bootstrap_results = np.zeros([N, 1])
        for i in range(N):
            bootstrap_results[i, :] = self.get_1cat_energy_ratio(
                use_observed_freq=use_observed_freq, randomize_df=True
            )

        # Return the results in the order used in previous versions
        results_array = np.array(
            [
                results[0],
                np.nanpercentile(bootstrap_results[:, 0], percentiles)[0],
                np.nanpercentile(bootstrap_results[:, 0], percentiles)[1],
            ]
        )
        return results_array

    def get_2cat_energy_ratio(self, use_observed_freq=True, randomize_df=False):

        results = self.get_energy_ratio(
            use_observed_freq=use_observed_freq, randomize_df=randomize_df
        )

        # Return with the difference as a convience
        return np.array(
            [
                results[0],
                results[1],
                results[1] - results[0],
                100 * (results[1] - results[0]) / results[0],
            ]
        )

    def get_2cat_energy_ratio_with_range(
        self, use_observed_freq=True, N=100, percentiles=[10, 90]
    ):

        # Get central results
        results = self.get_2cat_energy_ratio(
            use_observed_freq=use_observed_freq, randomize_df=False
        )

        # Get a bootstrap sample of range
        bootstrap_results = np.zeros([N, 4])
        for i in range(N):
            bootstrap_results[i, :] = self.get_2cat_energy_ratio(
                use_observed_freq=use_observed_freq, randomize_df=True
            )

        # Return the results in the order used in previous versions
        results_array = np.array(
            [
                results[0],
                np.nanpercentile(bootstrap_results[:, 0], percentiles)[0],
                np.nanpercentile(bootstrap_results[:, 0], percentiles)[1],
                results[1],
                np.nanpercentile(bootstrap_results[:, 1], percentiles)[0],
                np.nanpercentile(bootstrap_results[:, 1], percentiles)[1],
                results[2],
                np.nanpercentile(bootstrap_results[:, 2], percentiles)[0],
                np.nanpercentile(bootstrap_results[:, 2], percentiles)[1],
                results[3],
                np.nanpercentile(bootstrap_results[:, 3], percentiles)[0],
                np.nanpercentile(bootstrap_results[:, 3], percentiles)[1],
            ]
        )

        return results_array
