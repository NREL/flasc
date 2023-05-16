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
from pandas.errors import DataError
from scipy.interpolate import NearestNDInterpolator

from ..energy_ratio import energy_ratio as er
from ..energy_ratio import energy_ratio_gain as erg
from ..energy_ratio import energy_ratio_visualization as vis
from .. import time_operations as fsato, utilities as fsut


class energy_ratio_suite:
    """Wrapping class that relies internally on the energy_ratio class to
    import one or multiple datasets, mask data to specific atmospheric
    conditions, calculate the respective energy ratios, and plot everything.
    Essentially, this class facilitates easy comparisons between SCADA data
    and the FLORIS model. Furthermore, a common use case for this class
    is comparing the energy ratios with and without wake steering in field
    campaigns.
    """

    def __init__(self, verbose=False):
        """Initialization of the class. This creates several empty variables
        which can be populated using functions such as add_df.

        Args:
            verbose (bool, optional): Print to console. Defaults to False.
        """
        self.df_list = []
        self.num_turbines = []
        self.turbine_names = []

        # Placeholders for masks
        self.wd_range = None
        self.wd_range_ids = []
        self.ws_range = None
        self.ws_range_ids = []
        self.ti_range = None
        self.ti_range_ids = []
        self.time_range = None
        self.time_range_ids = []
        self.verbose = verbose

        if verbose:
            print("Initialized energy_ratio_suite() object.")

    # Public methods

    def add_df(self, df, name, color=None):
        """Add a dataframe to the class. This function verifies if the
        dataframe inserted matches in formatting with the already existing
        dataframes in the class. It also verifies if the right columns
        exist.

        Args:
            df_in ([pd.DataFrame]): The dataframe provided by the user. This
            dataframe should have the following columns:
                * Reference wind direction for the test turbine, 'wd'
                * Reference wind speed for the test turbine, 'ws'
                * Power production of every turbine: pow_000, pow_001, ...
                * Reference power production used to normalize the energy
                    ratio: 'pow_ref'
            name ([str]): Label for the dataframe.

        Raises:
            ImportError: This error is raised if the user-provided dataframe
                does not contain all necessary columns.
        """
        if not ("wd" in df.columns and "ws" in df.columns):
            raise ImportError(
                "Could not find all columns. Ensure that"
                + "you have columns 'wd' and 'ws' in your df."
            )

        num_turbines = fsut.get_num_turbines(df)
        if len(self.df_list) < 1:
            self.num_turbines = num_turbines
            self.turbine_names = [str(i) for i in range(num_turbines)]

        if self.num_turbines != num_turbines:
            raise UserWarning(
                "Added dataframe seems to have a different number of "
                + "turbines than the existing dataframe(s). Skipping "
                + "addition."
            )

        new_entry = dict({"df": df, "name": name, "color":color})
        self.df_list.append(new_entry)

        default_ids = np.array([True for _ in range(df.shape[0])])
        self.wd_range_ids.append(default_ids)
        self.ws_range_ids.append(default_ids)
        self.ti_range_ids.append(default_ids)
        self.time_range_ids.append(default_ids)

        # Force update mask for new dataframe
        idx = len(self.df_list) - 1
        wd_range_tmp = self.wd_range
        ws_range_tmp = self.ws_range
        ti_range_tmp = self.ti_range
        time_range_tmp = self.time_range
        self.wd_range = None
        self.ws_range = None
        self.ti_range = None
        self.time_range = None
        self.set_masks(
            ws_range=ws_range_tmp,
            wd_range=wd_range_tmp,
            ti_range=ti_range_tmp,
            time_range=time_range_tmp,
            df_ids=[idx],
        )

    def remove_df(self, index):
        """Remove a dataframe from the class.

        Args:
            index ([int]): Index of the to-be-removed dataframe.
        """
        if self.verbose:
            print(
                "Removing dataframe with name '"
                + self.df_list[index]["name"]
                + "'."
            )

        # Remove dataframe
        self.df_list.pop(index)

        # Remove mask indices for this dataframe
        self.wd_range_ids.pop(index)
        self.ws_range_ids.pop(index)
        self.ti_range_ids.pop(index)
        self.time_range_ids.pop(index)

        if len(self.df_list) < 1:
            # Reset variables
            self.num_turbines = []

    def print_dfs(self):
        """Print an overview of the contents and settings of all the
        dataframes currently in the class."""
        for ii in range(len(self.df_list)):
            print("___ DATAFRAME %d ___" % ii)
            keys = [c for c in self.df_list[ii].keys()]
            for c in keys:
                var = self.df_list[ii][c]
                if isinstance(var, pd.DataFrame):
                    print(
                        "  ["
                        + str(ii)
                        + "] "
                        + c
                        + ": "
                        + "pd.Dataframe() with shape ",
                        var.shape,
                    )
                else:
                    print("  [" + str(ii) + "] " + c + ": ", var)
            print(" ")

    def set_masks(
        self,
        ws_range=None,
        wd_range=None,
        ti_range=None,
        time_range=None,
        df_ids=None,
    ):
        """Mask all dataframes to a certain subset of data. One can
        mask data based on the wind speed (ws_range), wind direction
        (wd_range), turbulence intensity (ti_range), time (time_range)
        for all dataframes of interest. The dataframes of interest
        are assigned by their index, gathered a a list in df_ids.

        Args:
            ws_range ([iterable], optional): Wind speed mask. Should be an
                iterable of length 2, e.g., [6.0, 10.0], defining the lower
                and upper bound, respectively. If not specified, will not
                mask the data based on this variable. Defaults to None.
            wd_range ([iterable], optional): Wind direction mask. Should
                be an iterable of length 2, e.g., [0.0, 180.0], defining
                the lower and upper bound, respectively. If not specified,
                will not mask the data based on this variable. Defaults to
                None.
            ti_range ([iterable], optional): Turbulence intensity mask.
                Should be an iterable of length 2, e.g., [0.04, 0.08],
                defining the lower and upper bound, respectively. If not
                specified, will not mask the data based on this variable.
                Defaults to None.
            time_range ([iterable], optional): Wind speed mask. Should be an
                iterable of length 2, e.g., [pd.to_datetime("2019-01-01"),
                pd.to_datetime("2019-04-01")], defining the lower and upper
                bound, respectively. If not specified, will not mask the data
                based on this variable. Defaults to None.
            df_ids ([iterable], optional): List of turbine indices depicting
                which dataframes should be masked based on the specified
                criteria. If not specified, will apply the masks to all
                datasets. Defaults to None.
        """
        if self.verbose:
            print("Extracting a mask over the df: 'df_subset'.")

        if df_ids is None:
            df_ids = range(len(self.df_list))
        elif isinstance(df_ids, (int, np.integer, float)):
            df_ids = [int(df_ids)]

        if (ws_range is not None) and not (ws_range == self.ws_range):
            self.ws_range = ws_range
            for ii in df_ids:
                df = self.df_list[ii]["df"]
                ids = (df["ws"] > ws_range[0]) & (df["ws"] <= ws_range[1])
                self.ws_range_ids[ii] = np.array(ids)

        if (wd_range is not None) and not (wd_range == self.wd_range):
            self.wd_range = wd_range
            for ii in df_ids:
                df = self.df_list[ii]["df"]
                ids = (df["wd"] > wd_range[0]) & (df["wd"] <= wd_range[1])
                self.wd_range_ids[ii] = np.array(ids)

        if (ti_range is not None) and not (ti_range == self.ti_range):
            self.ti_range = ti_range
            for ii in df_ids:
                df = self.df_list[ii]["df"]
                ids = (df["ti"] > ti_range[0]) & (df["ti"] <= ti_range[1])
                self.ti_range_ids[ii] = np.array(ids)

        if (time_range is not None) and not (time_range == self.time_range):
            self.time_range = time_range
            for ii in df_ids:
                df = self.df_list[ii]["df"]
                ids = np.array([False for _ in range(df.shape[0])])
                indices_out = fsato.find_window_in_time_array(
                    df["time"], seek_time_windows=[list(time_range)]
                )
                ids[indices_out[0]] = True
                self.time_range_ids[ii] = ids

        # Update masked dataframe(s)
        for ii in df_ids:
            mask = (
                (self.wd_range_ids[ii])
                & (self.ws_range_ids[ii])
                & (self.ti_range_ids[ii])
                & (self.time_range_ids[ii])
            )
            df_full = self.df_list[ii]["df"]
            self.df_list[ii]["df_subset"] = df_full[mask]

    def set_turbine_names(self, turbine_names):
        """Assign turbine names/labels instead of just their index number.
        This can be useful when working with SCADA data where turbines are
        marked by a non-integer label.

        Args:
            turbine_names ([iterable]): List containing the turbine name
                strings.

        Raises:
            DataError: Will raise a DataError if the length of turbine_names
                does not match the number of turbines in the dataframe.
        """
        if not len(turbine_names) == self.num_turbines:
            raise DataError(
                "The length of turbine_names is incorrect."
                "Length should  be %d (specified: %d)."
                % (self.num_turbines, len(turbine_names))
            )
        self.turbine_names = turbine_names

    def clear_energy_ratio_results_of_dataset(self, ii):
        """Clear the energy ratio results for the ii'th dataset in the
        class.

        Args:
            ii ([int]): Dataset number/identifier
        """
        self.df_list[ii].pop("er_results")
        self.df_list[ii].pop("df_freq")
        self.df_list[ii].pop("er_test_turbines")
        self.df_list[ii].pop("er_ref_turbines")
        self.df_list[ii].pop("er_dep_turbines")
        self.df_list[ii].pop("er_wd_step")
        self.df_list[ii].pop("er_ws_step")
        self.df_list[ii].pop("er_wd_bin_width")
        self.df_list[ii].pop("er_bootstrap_N")

    def clear_energy_ratio_results_all(self):
        """Clear the energy ratio results for all datasets."""
        for ii in range(len(self.df_list)):
            self.clear_energy_ratio_results_of_dataset(ii)

    def get_energy_ratios(
        self,
        test_turbines,
        wd_step=3.0,
        ws_step=5.0,
        wd_bin_width=None,
        ws_bins=None,
        wd_bins=None,
        N=1,
        percentiles=[5.0, 95.0],
        balance_bins_between_dfs=True,
        return_detailed_output=False,
        num_blocks=-1,
        verbose=True,
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
                bin width can be specified separately. Note that this variable
                is ignored if the 'wd_bins' is also specified. Defaults to
                3.0.
            ws_step (float, optional): Wind speed discretization step size.
                This defines the resolution and widths of the wind speed
                bins. Note that this variable is ignored if the 'ws_bins' is
                also specified. Defaults to 5.0.
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
                currently. This variable overwrites the settings for 'ws_step'
                and instead allows the user to directly specify the binning
                properties, rather than deriving them from the data and an
                assigned step size. Defaults to None.
            wd_bins (array, optional): Array containing the bins over which
                the energy ratios must be calculated (wind dir.). Each entry
                of the provided array must contain exactly two float values,
                being the lower and upper bound for that wind dir. bin.
                Overlap between bins is supported for wind direction bins.
                This variable overwrites the settings for 'wd_step' and
                'wd_bin_width', and instead allows the user to directly
                specify the binning properties, rather than deriving them
                from the data and an assigned step size and bin width.
                Defaults to None.
            N (int, optional): Number of bootstrap evaluations for
                uncertainty quantification (UQ). If N=1, will not perform
                any uncertainty quantification. Defaults to 1.
            percentiles (list, optional): Confidence bounds for the
                uncertainty quantification in percents. This value is only
                relevant if N > 1 is specified. Defaults to [5., 95.].
            balance_bins_between_dfs (bool, optional): Balance the bins by
                the frequency of occurrence for each wind direction and wind
                speed bin in the collective of dataframes. Frequency of a
                certain bin is equal to the minimum number of occurrences
                among all the dataframes. This ensures we are comparing
                apples to apples. Recommended to set to 'True'. It will
                avoid bin rebalancing if the underlying wd and ws occurrences
                are identical between all dataframes, e.g., when we are
                comparing SCADA data to FLORIS predictions of the same data.
                Defaults to True.
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
            verbose (bool, optional): Print to console. Defaults to True.

        Returns:
            self.df_list (iterable): List of Pandas DataFrames containing
                the energy ratios for each dataset, respectively. Each
                entry in this list is a Dataframe containing the found
                energy ratios under the prespecified settings, contains the
                columns:
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
        # Define number of dataframes specified by user
        N_df = len(self.df_list)

        # Load energy ratio class for dfs without bin frequency interpolant
        era_list = [None for _ in range(N_df)]
        for ii in range(N_df):
            df_subset = self.df_list[ii]["df_subset"]
            era_list[ii] = er.energy_ratio(df_in=df_subset, verbose=verbose)

        if balance_bins_between_dfs:
            # First check if necessary
            balance_bins_between_dfs = False
            wd_ref = np.array(self.df_list[0]["df_subset"]["wd"])
            ws_ref = np.array(self.df_list[0]["df_subset"]["ws"])
            for d in self.df_list:
                if (
                    (not np.array_equal(wd_ref, d["df_subset"]["wd"])) or
                    (not np.array_equal(ws_ref, d["df_subset"]["ws"]))
                ):
                    balance_bins_between_dfs = True

            if balance_bins_between_dfs:
                print("Dataframes differ in wd and ws. Rebalancing.")
                df_binned_list = [None for _ in range(N_df)]
                for ii, era in enumerate(era_list):
                    # Calculate how data would be binned in era
                    era._set_test_turbines(test_turbines)
                    era._set_binning_properties(
                        ws_step=ws_step,
                        wd_step=wd_step,
                        wd_bin_width=wd_bin_width,
                        ws_bins=ws_bins,
                        wd_bins=wd_bins,
                    )
                    era._calculate_bins()

                    # Extract dataframe and calculate bin counts
                    df_binned = era.df[["wd_bin", "ws_bin"]].copy()
                    df_binned["bin_count_df{:d}".format(ii)] = 1
                    df_binned = df_binned.groupby(["wd_bin", "ws_bin"]).sum()
                    df_binned_list[ii] = df_binned

                # Now merge bin counts from each separate dataframe
                df_binned_merged = pd.concat(df_binned_list, axis=1)
                df_binned_merged = df_binned_merged.fillna(0).astype(int)

                # Determine minimum bin count for every ws/wd
                df_binned_merged["bin_count_balanced"] = (
                    df_binned_merged.min(axis=1)
                )

                # Define a bin frequency interpolant. Can be nearest-neighbor
                # since every data point from all dataframes is covered.
                df_binned_merged = df_binned_merged.reset_index(drop=False)
                freq_interpolant = NearestNDInterpolator(
                    x=df_binned_merged[["wd_bin", "ws_bin"]],
                    y=df_binned_merged["bin_count_balanced"],
                )

                # Assign frequency interpolant to each energy ratio object
                for era in era_list:
                    era._set_inflow_freq_interpolant(freq_interpolant)

            else:
                print(
                    "Dataframes share underlying wd and ws." +
                    " Skipping rebalancing -- not necessary."
                )

        # Now calculate energy ratios using each object
        for ii, era in enumerate(era_list):
            out = era.get_energy_ratio(
                test_turbines=test_turbines,
                wd_step=wd_step,
                ws_step=ws_step,
                wd_bin_width=wd_bin_width,
                ws_bins=ws_bins,
                wd_bins=wd_bins,
                N=N,
                percentiles=percentiles,
                return_detailed_output=return_detailed_output,
                num_blocks = num_blocks
            )

            # Save each output to self
            if return_detailed_output:
                self.df_list[ii]["er_results"] = out[0]
                self.df_list[ii]["er_results_info_dict"] = out[1]
            else:
                self.df_list[ii]["er_results"] = out
            self.df_list[ii]["df_freq"] = era.df_freq.reset_index(drop=False)
            self.df_list[ii]["er_test_turbines"] = test_turbines
            self.df_list[ii]["er_wd_step"] = wd_step
            self.df_list[ii]["er_ws_step"] = ws_step
            self.df_list[ii]["er_wd_bin_width"] = era.wd_bin_width
            self.df_list[ii]["er_bootstrap_N"] = N

        return self.df_list

    def get_energy_ratios_gain(
        self,
        test_turbines,
        wd_step=3.0,
        ws_step=5.0,
        wd_bin_width=None,
        ws_bins=None,
        wd_bins=None,
        N=1,
        percentiles=[5.0, 95.0],
        # balance_bins_between_dfs=True,
        return_detailed_output=False,
        num_blocks=-1,
        verbose=True,
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
                bin width can be specified separately. Note that this variable
                is ignored if the 'wd_bins' is also specified. Defaults to
                3.0.
            ws_step (float, optional): Wind speed discretization step size.
                This defines the resolution and widths of the wind speed
                bins. Note that this variable is ignored if the 'ws_bins' is
                also specified. Defaults to 5.0.
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
                currently. This variable overwrites the settings for 'ws_step'
                and instead allows the user to directly specify the binning
                properties, rather than deriving them from the data and an
                assigned step size. Defaults to None.
            wd_bins (array, optional): Array containing the bins over which
                the energy ratios must be calculated (wind dir.). Each entry
                of the provided array must contain exactly two float values,
                being the lower and upper bound for that wind dir. bin.
                Overlap between bins is supported for wind direction bins.
                This variable overwrites the settings for 'wd_step' and
                'wd_bin_width', and instead allows the user to directly
                specify the binning properties, rather than deriving them
                from the data and an assigned step size and bin width.
                Defaults to None.
            N (int, optional): Number of bootstrap evaluations for
                uncertainty quantification (UQ). If N=1, will not perform
                any uncertainty quantification. Defaults to 1.
            percentiles (list, optional): Confidence bounds for the
                uncertainty quantification in percents. This value is only
                relevant if N > 1 is specified. Defaults to [5., 95.].
            balance_bins_between_dfs (bool, optional): Balance the bins by
                the frequency of occurrence for each wind direction and wind
                speed bin in the collective of dataframes. Frequency of a
                certain bin is equal to the minimum number of occurrences
                among all the dataframes. This ensures we are comparing
                apples to apples. Recommended to set to 'True'. It will
                avoid bin rebalancing if the underlying wd and ws occurrences
                are identical between all dataframes, e.g., when we are
                comparing SCADA data to FLORIS predictions of the same data.
                Defaults to True.
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
            verbose (bool, optional): Print to console. Defaults to True.

        Returns:
            self.df_list (iterable): List of Pandas DataFrames containing
                the energy ratios for each dataset, respectively. Each
                entry in this list is a Dataframe containing the found
                energy ratios under the prespecified settings, contains the
                columns:
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

        #TODO should probably check that num df = 2,4,6 etc.,

        # Define number of dataframes specified by user
        N_df = len(self.df_list)
        N_gains = int(N_df / 2) # Assume every 2 form a desired gain

        # Set up a list of gains
        self.df_list_gains = []

        # Load energy ratio class for dfs without bin frequency interpolant
        era_list = [None for _ in range(N_gains)]
        for ii in range(0, N_df, 2):# (N_df, step=2):
            df_subset_d = self.df_list[ii]["df_subset"]
            df_subset_n = self.df_list[ii+1]["df_subset"]

            name_d = self.df_list[ii]["name"]
            name_n = self.df_list[ii+1]["name"]

            color = self.df_list[ii]["color"]

            era_list[int(ii/2)] = erg.energy_ratio_gain(df_in_d=df_subset_d, df_in_n=df_subset_n, verbose=verbose)

            new_entry = dict({"name": '%s/%s' % (name_n, name_d), "color":color})
            self.df_list_gains.append(new_entry)

        if True: # balance_bins_between_dfs: TODO: I think this is a must in this case
            # First check if necessary
            balance_bins_between_dfs = False
            wd_ref = np.array(self.df_list[0]["df_subset"]["wd"])
            ws_ref = np.array(self.df_list[0]["df_subset"]["ws"])
            for d in self.df_list:
                if (
                    (not np.array_equal(wd_ref, d["df_subset"]["wd"])) or
                    (not np.array_equal(ws_ref, d["df_subset"]["ws"]))
                ):
                    balance_bins_between_dfs = True

            if True: #balance_bins_between_dfs: #TODO Again just forcing this here I think
                print("Dataframes differ in wd and ws. Rebalancing.")
                df_binned_list = [None for _ in range(N_df)]
                for ii, era in enumerate(era_list):
                    # Calculate how data would be binned in era
                    era._set_test_turbines(test_turbines)
                    era._set_binning_properties(
                        ws_step=ws_step,
                        wd_step=wd_step,
                        wd_bin_width=wd_bin_width,
                        ws_bins=ws_bins,
                        wd_bins=wd_bins,
                    )
                    era._calculate_bins()

                    # Extract dataframe and calculate bin counts
                    # Do this for both _d and _n
                    df_binned = era.df_d[["wd_bin", "ws_bin"]].copy()
                    df_binned["bin_count_df{:d}".format(ii*2)] = 1
                    df_binned = df_binned.groupby(["wd_bin", "ws_bin"]).sum()
                    df_binned_list[ii*2] = df_binned

                    df_binned = era.df_n[["wd_bin", "ws_bin"]].copy()
                    df_binned["bin_count_df{:d}".format(ii*2 + 1)] = 1
                    df_binned = df_binned.groupby(["wd_bin", "ws_bin"]).sum()
                    df_binned_list[ii*2 + 1] = df_binned

                # Now merge bin counts from each separate dataframe
                df_binned_merged = pd.concat(df_binned_list, axis=1)
                df_binned_merged = df_binned_merged.fillna(0).astype(int)

                # Determine minimum bin count for every ws/wd
                df_binned_merged["bin_count_balanced"] = (
                    df_binned_merged.min(axis=1)
                )

                # Define a bin frequency interpolant. Can be nearest-neighbor
                # since every data point from all dataframes is covered.
                df_binned_merged = df_binned_merged.reset_index(drop=False)
                freq_interpolant = NearestNDInterpolator(
                    x=df_binned_merged[["wd_bin", "ws_bin"]],
                    y=df_binned_merged["bin_count_balanced"],
                )

                # Assign frequency interpolant to each energy ratio object
                for era in era_list:
                    era._set_inflow_freq_interpolant(freq_interpolant)

            else:
                print(
                    "Dataframes share underlying wd and ws." +
                    " Skipping rebalancing -- not necessary."
                )

        # Now calculate energy ratios using each object
        for ii, era in enumerate(era_list):
            out = era.get_energy_ratio_gain(
                test_turbines=test_turbines,
                wd_step=wd_step,
                ws_step=ws_step,
                wd_bin_width=wd_bin_width,
                ws_bins=ws_bins,
                wd_bins=wd_bins,
                N=N,
                percentiles=percentiles,
                return_detailed_output=return_detailed_output,
                num_blocks = num_blocks
            )

            # Save each output to self
            if return_detailed_output:
                self.df_list_gains[ii]["er_results"] = out[0]
                self.df_list_gains[ii]["er_results_info_dict"] = out[1]
            else:
                self.df_list_gains[ii]["er_results"] = out
            self.df_list_gains[ii]["df_freq"] = era.df_freq.reset_index(drop=False)
            self.df_list_gains[ii]["er_test_turbines"] = test_turbines
            self.df_list_gains[ii]["er_wd_step"] = wd_step
            self.df_list_gains[ii]["er_ws_step"] = ws_step
            self.df_list_gains[ii]["er_wd_bin_width"] = era.wd_bin_width
            self.df_list_gains[ii]["er_bootstrap_N"] = N

        return self.df_list_gains

    def get_energy_ratios_fast(
        self,
        test_turbines,
        wd_step=3.0,
        ws_step=5.0,
        wd_bin_width=None,
        ws_bins=None,
        wd_bins=None,
        verbose=True,
    ):
        """This is a fast function surpassing several additional steps that
        take place in get_energy_ratios() which are time consuming but are
        not essential to the calculation of the energy ratio itself, under
        certain conditions:
          1) No bootstrapping
          2) The data needs not be reweighted, i.e., the underlying wind
             rose and data occurrences are identical between all datasets
             in this class. This is always the case when you are comparing
             SCADA to its FLORIS prediction counterparts, but definitely
             not the case when comparing different SCADA datasets.

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
                currently. This variable overwrites the settings for 'ws_step'
                and instead allows the user to directly specify the binning
                properties, rather than deriving them from the data and an
                assigned step size. Defaults to None.
            wd_bins (array, optional): Array containing the bins over which
                the energy ratios must be calculated (wind dir.). Each entry
                of the provided array must contain exactly two float values,
                being the lower and upper bound for that wind dir. bin.
                Overlap between bins is supported for wind direction bins.
                This variable overwrites the settings for 'wd_step' and
                'wd_bin_width', and instead allows the user to directly
                specify the binning properties, rather than deriving them
                from the data and an assigned step size and bin width.
                Defaults to None.
            verbose (bool, optional): Print to console. Defaults to True.

        Returns:
            self.df_list (iterable): List of Pandas DataFrames containing
                the energy ratios for each dataset, respectively. Each
                entry in this list is a Dataframe containing the found
                energy ratios under the prespecified settings, contains the
                columns:
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
        # Define number of dataframes specified by user
        N_df = len(self.df_list)

        # Load energy ratio class for dfs without bin frequency interpolant
        era_list = [None for _ in range(N_df)]
        for ii in range(N_df):
            df_subset = self.df_list[ii]["df_subset"]
            era_list[ii] = er.energy_ratio(df_in=df_subset, verbose=verbose)

        # Now calculate energy ratios using fast method for each object
        for ii, era in enumerate(era_list):
            out = era.get_energy_ratio_fast(
                test_turbines=test_turbines,
                wd_step=wd_step,
                ws_step=ws_step,
                wd_bin_width=wd_bin_width,
                ws_bins=ws_bins,
                wd_bins=wd_bins,
            )

            # Save each output to self
            self.df_list[ii]["er_results"] = out
            self.df_list[ii]["df_freq"] = None
            self.df_list[ii]["er_test_turbines"] = test_turbines
            self.df_list[ii]["er_wd_step"] = wd_step
            self.df_list[ii]["er_ws_step"] = ws_step
            self.df_list[ii]["er_wd_bin_width"] = era.wd_bin_width
            self.df_list[ii]["er_bootstrap_N"] = 1

        return self.df_list

    def plot_energy_ratios(self, 
                           superimpose=True, 
                           hide_uq_labels=True, 
                           polar_plot=False, 
                           axarr=None,
                           show_barplot_legend=True):
        """This function plots the energy ratios of each dataset against
        the wind direction, potentially with uncertainty bounds if N > 1
        was specified by the user. One must first run get_energy_ratios()
        before attempting to plot the energy ratios.

        Args:
        superimpose (bool, optional): if True, plots the energy ratio
            of all datasets into the same figure. If False, will plot the
            energy ratio of each dataset into a separate figure. Defaults
            to True.
        hide_uq_labels (bool, optional): If true, do not specifically label
            the confidence intervals in the plot
        polar_plot (bool, optional): Plots the energy ratios in a polar
            coordinate system, aligned with the wind direction coordinate
            system of FLORIS. Defaults to False.
        axarr([iteratible]): List of axes in the figure with length 2.
        show_barplot_legend (bool, optional): Show the legend in the bar
            plot figure?  Defaults to True

        Returns:
            axarr([iteratible]): List of axes in the figure with length 2.
        """
        if superimpose:
            results_array = [df["er_results"] for df in self.df_list]
            df_freq_array = [df["df_freq"] for df in self.df_list]
            labels_array = [df["name"] for df in self.df_list]
            colors_array = [df["color"] for df in self.df_list]
            axarr = vis.plot(
                energy_ratios=results_array,
                df_freqs=df_freq_array,
                labels=labels_array,
                colors=colors_array,
                hide_uq_labels=hide_uq_labels,
                polar_plot=polar_plot,
                axarr=axarr,
                show_barplot_legend=show_barplot_legend
            )

        else:
            # If superimpose is False, axarr must be None
            if axarr is not None:
                raise ValueError("If superimpose is False, axarr must be None")

            axarr = []
            for df in self.df_list:
                axi = vis.plot(
                    energy_ratios=df["er_results"],
                    df_freqs=df["df_freq"],
                    labels=df["name"],
                    colors=[df["color"]],
                    hide_uq_labels=hide_uq_labels,
                    polar_plot=polar_plot,
                    show_barplot_legend=show_barplot_legend
                )
                axarr.append(axi)

        return axarr

    def plot_energy_ratio_gains(
            self, 
            superimpose=True, 
            hide_uq_labels=True,
            axarr=None,
            show_barplot_legend=True,
    ):
        """This function plots the energy ratios of each dataset against
        the wind direction, potentially with uncertainty bounds if N > 1
        was specified by the user. One must first run get_energy_ratios()
        before attempting to plot the energy ratios.

        Args:
            superimpose (bool, optional): if True, plots the energy ratio
            of all datasets into the same figure. If False, will plot the
            energy ratio of each dataset into a separate figure. Defaults
            to True.
            hide_uq_labels (bool, optional): If true, do not specifically label
            the confidence intervals in the plot
            axarr([iteratible]): List of axes in the figure with length 2.
            show_barplot_legend (bool, optional): Show the legend in the bar
            plot figure?  Defaults to True

        Returns:
            axarr([iteratible]): List of axes in the figure with length 2.
        """
        if superimpose:
            results_array = [df["er_results"] for df in self.df_list_gains]
            labels_array = [df["name"] for df in self.df_list_gains]
            colors_array = [df["color"] for df in self.df_list_gains]
            axarr = vis.plot(
                energy_ratios=results_array, 
                labels=labels_array, 
                colors=colors_array,
                hide_uq_labels=hide_uq_labels,
                axarr=axarr,
                show_barplot_legend=show_barplot_legend
            )
            axarr[0].set_ylabel("Change in energy ratio (-)")

        else:
            # If superimpose is False, axarr must be None
            if axarr is not None:
                raise ValueError("If superimpose is False, axarr must be None")
            axarr = []
            for df in self.df_list_gains:
                axi = vis.plot(
                    energy_ratios=df["er_results"],
                    labels=df["name"], 
                    colors=[df["color"]],
                    hide_uq_labels=hide_uq_labels,
                    axarr=axarr,
                    show_barplot_legend=show_barplot_legend
                )
                axi.set_ylabel("Change in energy ratio (-)")
                axarr.append(axi)

        return axarr


    def export_detailed_energy_info_to_xlsx(
        self,
        fout_xlsx,
        hide_bin_count_columns=False,
        hide_ws_ti_columns=False,
        hide_pow_columns=False,
        hide_unbalanced_cols=True,
        fi=None
    ):
        """This function will calculate an energy table saved to excel.

        Args:
            test_turbines ([iteratible]): List with the test turbine(s)
                which will be calculated in the spreadsheet
            wd_bins (np.array): Wind direction bins to analyze
            ws_bins (np.array): Wind speed bins to analyze
            fout_xlsx (str): The path and filename to which the .xlsx
                excel file will be saved.
            fi ([type], optional): FLORIS object for the wind farm that can
                be used to generate visuals of wind direction within the
                sheets. Defaults to None.
        """
        # Check if detailed information available. If not, return error.
        if not ("er_results_info_dict" in self.df_list[0].keys()):
            raise DataError(
                "Did not find detailed energy ratio information. " +
                "Make sure you run .get_energy_ratios() with "
                " `return_detailed_output=True` before calling this function."
            )

        # # Build the table, assuming that we don't need to balance
        vis.table_analysis(
            df_list=self.df_list,
            fout_xlsx=fout_xlsx,
            hide_bin_count_columns=hide_bin_count_columns,
            hide_ws_ti_columns=hide_ws_ti_columns,
            hide_pow_columns=hide_pow_columns,
            hide_unbalanced_cols=hide_unbalanced_cols,
            fi=fi,
        )
