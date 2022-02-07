# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from operational_analysis.toolkits import filters

from .. import time_operations as fsato, utilities as fsut
from ..dataframe_operations import dataframe_filtering as dff
from ..turbine_analysis import ws_pow_filtering_utilities as ut




class ws_pw_curve_filtering:
    """This class allows a user to filter turbine data based on the
    wind-speed power curve. This class includes several useful filtering
    methods:
        1. Filtering based on prespecified boxes/windows. Any data outside
           of the specified box is considered faulty.
        2. Filtering based on x-wise distance from the mean power curve. Any
           data too far off the mean curve is considered faulty.
        3. Filtering based on the standard deviation from the mean power
           curve. This is slightly different from (2) in the point that
           it allows the user to consider variations in standard deviation
           per power bin.
    """
    def __init__(
        self,
        df,
        turbine_list="all",
        add_default_windows=False,
        rated_powers=None,
    ):
        """Initializes the class.

        Args:
            df ([pd.DataFrame]): Dataframe containing the turbine data,
                formatted in the generic SCADA data format. Namely, the
                dataframe should at the very least contain the columns:
                  * Time of each measurement: time
                  * Wind speed of each turbine: ws_000, ws_001, ... 
                  * Power production of each turbine: pow_000, pow_001, ...
            turbine_list (iteratible, optional): List with turbine numbers
                that should be filtered for. If "all" is specified, then
                it will automatically determine the number of turbines and
                assign turbine_list as range(num_turbs). Defaults to "all".
            add_default_windows (bool, optional): Add default filtering
                windows (filter method 1) based on the (estimated) rated
                power of every turbine. Defaults to False.
            rated_powers ([iteratible], optional): List with the rated
                power production for every turbine. If only a single float
                is provided it will assume all turbines have that value's
                rated power. If left unspecified, the rated power for
                every turbine will be derived from the provided data.
                Defaults to None.
        """
        self._set_df(df)
        self._set_turbine_mode(turbine_list, initialization=True)

        # Assign rated power for every turbine
        if rated_powers is None:
            self.rated_powers = ut.estimate_rated_powers_from_data(self.df)
        else:
            if isinstance(rated_powers, (int, float, np.integer, np.float64)):
                self.rated_powers = np.full(self.nturbs_all, rated_powers)
            else:
                self.rated_powers = np.array(rated_powers, dtype=float)

        # Prepare window filtering variables and defaults
        self.window_remove_all()  # Initialize empty windows array
        if add_default_windows:
            self._add_default_windows()

        # Initialize empty variables for filters
        self.reset_filters()

    # Private methods
    def _set_df(self, df):
        """Format and save the provided dataframe to the class.

        Args:
            df ([pd.DataFrame]): Dataframe containing the turbine data,
                formatted in the generic SCADA data format. Namely, the
                dataframe should at the very least contain the columns:
                  * Time of each measurement: time
                  * Wind speed of each turbine: ws_000, ws_001, ... 
                  * Power production of each turbine: pow_000, pow_001, ...
        """
        self.df = df.reset_index(drop=("time" in df.columns))
        self.dt = fsut.estimate_dt(self.df["time"])
        self.nturbs_all = fsut.get_num_turbines(df)
        self.full_turbs_list = range(self.nturbs_all)

    def _set_turbine_mode(self, turbine_list, initialization=False):
        """Assign which turbine(s) should be considered for filtering and
        plotting.

        Args:
            turbine_list (iteratible, optional): List with turbine numbers
                that should be filtered for. If "all" is specified, then
                it will automatically determine the number of turbines and
                assign turbine_list as range(num_turbs). Defaults to "all".
        """
        if isinstance(turbine_list, str):
            if turbine_list == "all":
                num_turbines = fsut.get_num_turbines(self.df)
                turbine_list = range(num_turbines)
            else:
                raise KeyError("Invalid turbine_list specified.")

        self.turbine_list = turbine_list
        self.num_turbines = len(turbine_list)
        if not initialization:
            self._get_mean_power_curves()

    def _add_default_windows(self):
        """Adds two windows to filter over based on the (estimated) rated
        power production of each turbine."""
        # First figure out which turbines can be clumped together (same rated pow.)
        turbs_sorted = []
        ratedpwrs = np.unique(self.rated_powers)
        for ii in range(len(ratedpwrs)):
            turbs = np.where(np.array(self.rated_powers) == ratedpwrs[ii])[0]
            turbs = np.sort(turbs)
            if len(turbs) > 2:
                try_range = range(turbs[0], turbs[-1] + 1)
                if np.array_equal(np.array(try_range), turbs):
                    turbs = try_range
            print(
                "Estimated rated power of turbines %s in this dataset to be %.1f"
                % (str(ut.convert_list_to_ranges((turbs))), ratedpwrs[ii])
            )
            turbs_sorted.append(np.array(turbs))

        # Setup windows and binning properties
        for ii, turbs in enumerate(turbs_sorted):
            default_w0_ws = (0.0, 15.0)
            default_w0_pw = (0.0, 0.95 * ratedpwrs[ii])
            default_w1_ws = (0.0, 25.0)
            default_w1_pw = (0.0, 1.04 * ratedpwrs[ii])

            idx = len(self.window_list)
            print(
                "Adding window[%d] and window[%d] for turbines:"
                % (idx, idx + 1),
                ut.convert_list_to_ranges(turbs),
            )
            self.window_add(
                default_w0_ws, default_w0_pw, axis=0, turbines=turbs
            )
            self.window_add(
                default_w1_ws, default_w1_pw, axis=1, turbines=turbs
            )

    def _get_mean_power_curves(self, ws_bins=np.arange(0.0, 25.5, 0.5)):
        """Calculates the mean power production in bins of the wind speed.

        Args:
            ws_bins ([iteratible], optional): Wind speed bins. Defaults to
                np.arange(0.0, 25.5, 0.5).

        Returns:
            pw_curve_df ([pd.DataFrame]): Dataframe containing the wind
                speed bins and the mean power production value for every
                turbine in self.turbine_list.
        """
        ws_max = np.max(ws_bins)
        ws_min = np.min(ws_bins)
        pw_curve_df = pd.DataFrame(
            {
                "ws": (ws_bins[1::] + ws_bins[0:-1]) / 2,
                "ws_min": ws_bins[0:-1],
                "ws_max": ws_bins[1::],
            }
        )

        for ti in self.turbine_list:
            ws = self.df["ws_%03d" % ti]
            pow = self.df["pow_%03d" % ti]
            status = self.df_filters[ti]["status"]
            clean_ids = (status == 1) & (ws > ws_min) & (ws < ws_max)
            ws_clean = ws[clean_ids]
            pw_clean = pow[clean_ids]

            # bin_array = np.digitize(ws_clean, ws_bins_l, right=False)
            bin_array = np.searchsorted(ws_bins, ws_clean, side="left")
            bin_array = bin_array - 1  # 0 -> 1st bin, rather than before bin
            pow_bins = [
                np.median(pw_clean[bin_array == i])
                for i in range(pw_curve_df.shape[0])
            ]

            # Write outputs to the dataframe
            pw_curve_df["pow_%03d" % ti] = pow_bins
            self.pw_curve_df = pw_curve_df

        return pw_curve_df

    def _update_status_flags(self, verbose=True):
        """Update the status flags based on the filtering choices made.
        The status flags are part of the self.df_filters dataframe which
        contains the information on which data points are marked faulty
        and by what filter(s)."""
        for df_f in self.df_filters:
            cols = [c for c in df_f.columns if "status" not in c]
            df_f["status"] = ~df_f[cols].any(axis=1)

        if verbose:
            N = self.df.shape[0]
            for ti in self.turbine_list:
                df_f = self.df_filters[ti]

                print("  Turbine %03d:" % ti)
                Nc = df_f["status"].sum()
                print("    Clean data: %d (%.3f%%)." % (Nc, 100.0 * Nc / N))

                for c in cols:
                    Nf = df_f[c].sum()
                    print("    %s: %d (%.3f%%)." % (c, Nf, 100.0 * Nf / N))

        self._get_mean_power_curves()  # Update mean power curve

    # Public methods
    def reset_filters(self):
        """Reset all filter variables and assume all data is clean."""
        # Reset certain variables
        self.pw_curve_df_bounds = None

        # Reset filtering bool arrays
        self.df_filters = [None for _ in range(self.nturbs_all)]
        for ti in range(self.nturbs_all):
            self.df_filters[ti] = pd.DataFrame(
                {
                    "is_nan": np.isnan(self.df["pow_%03d" % ti]),
                    "window_outlier": [False] * self.df.shape[0],
                    "ws_std_dev_outlier": [False] * self.df.shape[0],
                    "mean_pow_curve_outlier": [False] * self.df.shape[0],
                }
            )
        self._update_status_flags(verbose=False)

    def window_add(self, ws_range, pow_range, axis=0, turbines="all"):
        """Add a filtering window for all or a particular set of turbines.
        Any data that falls outside of this window will be removed, either
        along the x-axis (wind speed, axis = 0) or along the y-axis
        (power, axis = 1).

        Args:
            ws_range ([list, tuple]): Wind speed range in which data is OK.

            pow_range ([list, tuple]): Power measurement range in which data
            is OK.

            axis (int, optional): Specify the axis over which values outside
            of the window will be removed. axis=0 means limiting values lower
            and higher than the specified pow_range, within the ws_range.
            axis=1 means limiting values lower/higher than the ws_range
            and that fall within the pow_range. Defaults to 0.

            turbines (list, optional): Turbines to which this filter should
            apply. If unspecified, then it defaults to "all".
        """

        if isinstance(turbines, str):
            if turbines == "all":
                turbines = self.full_turbs_list
        elif isinstance(turbines, (int, np.integer)):
            turbines = [turbines]

        idx = len(self.window_list)
        new_entry = {
            "idx": idx,
            "ws_range": ws_range,
            "pow_range": pow_range,
            "axis": axis,
            "turbines": turbines,
        }
        self.window_list.append(new_entry)

    def window_remove(self, ids_to_remove):
        """Remove the specified filtering window.

        Args:
            ids_to_remove ([int]): Index of the window to remove
        """
        if not isinstance(ids_to_remove, (list, np.array)):
            ids_to_remove = [ids_to_remove]
        ids_to_remove = np.sort(ids_to_remove)[::-1]
        for i in ids_to_remove:
            self.window_list.pop(i)

        # Update indices
        for i in range(len(self.window_list)):
            self.window_list[i]["idx"] = i

    def window_remove_all(self):
        """Remove all filtering windows."""
        self.window_list = []

    def window_print_all(self):
        """Print information of all filter windows to console"""
        for i in range(len(self.window_list)):
            window = self.window_list[i]
            for k in window.keys():
                if k == "turbines":
                    str_short = ut.convert_list_to_ranges(
                        self.window_list[i][k]
                    )
                    print("window_list[%d][%s] = " % (i, k), str_short)
                elif not k == "idx":
                    print(
                        "window_list[%d][%s] = " % (i, k),
                        self.window_list[i][k],
                    )
            print("")

    def filter_by_windows(self):
        """Apply window filters to the dataset for the turbines of interest.
        """        
        print("Filtering data by specified regions...")
        for ti in self.turbine_list:
            df = self.df.copy()

            out_of_window_ids = np.zeros(df.shape[0])
            window_list = [w for w in self.window_list if ti in w["turbines"]]
            print(" ")
            print(
                "Applying %d window filters to the df for turbine %d"
                % (len(window_list), ti)
            )

            for window in window_list:
                idx = window["idx"]
                ws_range = window["ws_range"]
                pow_range = window["pow_range"]
                axis = window["axis"]
                if axis == 0:
                    ii_out_of_window = filters.window_range_flag(
                        df["pow_%03d" % ti],
                        pow_range[0],
                        pow_range[1],
                        df["ws_%03d" % ti],
                        ws_range[0],
                        ws_range[1],
                    )
                else:
                    ii_out_of_window = filters.window_range_flag(
                        df["ws_%03d" % ti],
                        ws_range[0],
                        ws_range[1],
                        df["pow_%03d" % ti],
                        pow_range[0],
                        pow_range[1],
                    )

                # Merge findings from all windows
                out_of_window_ids[ii_out_of_window] = int(1)
                print(
                    "  Removed %d outliers using window[%d]."
                    % (int(sum(ii_out_of_window)), idx)
                )

            print(
                "Removed a total of %d outliers using the %d windows."
                % (int(sum(out_of_window_ids)), len(window_list))
            )
            df_out_of_windows = np.zeros(self.df.shape[0])
            out_of_window_indices = df.index[np.where(out_of_window_ids)[0]]
            df_out_of_windows[out_of_window_indices] = 1
            self.df_filters[ti]["window_outlier"] = [
                bool(i) for i in df_out_of_windows
            ]

        # Finally, update status columns in dataframe
        self._update_status_flags()

    def filter_by_power_curve(
        self,
        m_ws_lb=0.95,
        m_pow_lb=1.01,
        m_ws_rb=1.05,
        m_pow_rb=0.99,
        ws_deadband=0.50,
        pow_deadband=20.0,
        no_iterations=10,
        cutoff_ws=25.0,
    ):
        """Filter the data by offset from the mean power curve in x-
        directions. This is an iterative process because the estimated mean
        curve actually changes as data is filtered. This process typically
        converges within a couple iterations.

        Args:
            m_ws_lb (float, optional): Multiplier on the wind speed defining
            the left bound for the power curve. Any data to the left of this
            curve is considered faulty. Defaults to 0.95.
            m_pow_lb (float, optional): Multiplier on the power defining
            the left bound for the power curve. Any data to the left of this
            curve is considered faulty. Defaults to 1.01.
            m_ws_rb (float, optional): Multiplier on the wind speed defining
            the right bound for the power curve. Any data to the right of this
            curve is considered faulty. Defaults to 1.05.
            m_pow_rb (float, optional): Multiplier on the power defining
            the right bound for the power curve. Any data to the right of this
            curve is considered faulty. Defaults to 0.99.
            no_iterations (int, optional): Number of iterations. The
            solution typically converges in 2-3 steps, but as the process is
            very fast, it's better to run a higher number of iterations.
            Defaults to 10.
        """
        print("Filtering data by deviations from the mean power curve...")
        for ii in range(no_iterations):
            # Create upper and lower bounds around mean curve
            df_xy = self.pw_curve_df.copy()
            x_full = np.array(df_xy["ws"], dtype=float)
            x = x_full[x_full < cutoff_ws]  # Only filter until 15 m/s
            self.pw_curve_df_bounds = pd.DataFrame({"ws": x})

            for ti in self.turbine_list:
                y = np.array(df_xy["pow_%03d" % ti], dtype=float)
                y = y[x_full < cutoff_ws]  # Only filter until 15 m/s
                if np.all(np.isnan(y)):
                    self.pw_curve_df_bounds["pow_%03d_lb" % ti] = None
                    self.pw_curve_df_bounds["pow_%03d_rb" % ti] = None
                    continue

                # Create interpolants to left and right of mean curve
                ws_array = np.array(self.df["ws_%03d" % ti], dtype=float)
                pow_array = np.array(self.df["pow_%03d" % ti], dtype=float)

                # Specify left side bound and non-decreasing
                lb_ws = x * m_ws_lb - ws_deadband / 2.0
                lb_pow = y * m_pow_lb + pow_deadband / 2.0

                # Make sure first couple entries are not NaN
                jjj = 0
                while np.isnan(lb_pow[jjj]):
                    lb_pow[jjj] = jjj / 1000.0
                    jjj = jjj + 1

                # Ensure non-decreasing for lower half of wind speeds
                id_center = np.argmin(np.abs(lb_ws - 9.0))  # Assume value is fine near 9 m/s
                lb_ws_l = lb_ws[0:id_center]
                lb_pow_l = lb_pow[0:id_center]
                good_ids = (
                    np.hstack([(np.diff(lb_pow_l) >= 0.0), True])
                    & 
                    (~np.isnan(lb_pow[0:id_center]))
                )
                good_ids[0] = True
                lb_pow_l = np.interp(lb_ws_l, lb_ws_l[good_ids], lb_pow_l[good_ids])
                lb_pow[0:id_center] = lb_pow_l
                non_nans = (~np.isnan(lb_pow) & ~np.isnan(lb_ws))
                lb_pow = lb_pow[non_nans]
                lb_ws = lb_ws[non_nans]

                # Specify right side bound and ensure monotonically increasing
                rb_ws = x * m_ws_rb + ws_deadband / 2.0
                rb_pow = y * m_pow_rb - pow_deadband / 2.0

                # Make sure first couple entries are not NaN
                jjj = 0
                while np.isnan(rb_pow[jjj]):
                    rb_pow[jjj] = jjj / 1000.0
                    jjj = jjj + 1

                # Ensure non-decreasing for lower half of wind speeds
                id_center = np.argmin(np.abs(rb_ws - 9.0))  # Assume value is fine near 9 m/s
                rb_ws_l = rb_ws[0:id_center]
                rb_pow_l = rb_pow[0:id_center]
                good_ids = (
                    np.hstack([(np.diff(rb_pow_l) >= 0.0), True])
                    & 
                    (~np.isnan(rb_pow[0:id_center]))
                )
                good_ids[0] = True
                rb_pow_l = np.interp(rb_ws_l, rb_ws_l[good_ids], rb_pow_l[good_ids])
                rb_pow[0:id_center] = rb_pow_l
                non_nans = (~np.isnan(rb_pow) & ~np.isnan(rb_ws))
                rb_pow = rb_pow[non_nans]
                rb_ws = rb_ws[non_nans]

                # Finally interpolate
                ws_lb = np.interp(
                    x=pow_array,
                    xp=lb_pow,
                    fp=lb_ws,
                    left=np.nan,
                    right=np.nan,
                )
                ws_rb = np.interp(
                    x=pow_array,
                    xp=rb_pow,
                    fp=rb_ws,
                    left=np.nan,
                    right=np.nan,
                )

                out_of_bounds = (ws_array < ws_lb) | (ws_array > ws_rb)
                self.df_filters[ti]["mean_pow_curve_outlier"] = out_of_bounds

                # Write left and right bound to own curve
                self.pw_curve_df_bounds["pow_%03d_lb" % ti] = np.interp(
                    x=x,
                    xp=lb_ws,
                    fp=lb_pow,
                    left=np.nan,
                    right=np.nan,
                )
                self.pw_curve_df_bounds["pow_%03d_rb" % ti] = np.interp(
                    x=x,
                    xp=rb_ws,
                    fp=rb_pow,
                    left=np.nan,
                    right=np.nan,
                )

            # Update status flags and re-estimate mean power curve
            verbose = ii == no_iterations - 1  # Only print final iteration
            self._update_status_flags(verbose=verbose)

    def filter_by_wsdev(
        self, pow_bin_width=20.0, max_ws_dev=2.0, pow_min=20.0, pow_max=None
    ):
        """Filter data that is too far off the mean curve w.r.t. the
        standard deviation in x-direction. This is slightly different from
        filtering by deviations from the mean curve as now the standard
        deviation of the data is taken into account.

        Args:
            pow_bin_width (float, optional): Bin width in the y-axis, thus
            over the power production. Defaults to 20.0.
            max_ws_dev (float, optional): Data points further than
            max_ws_dev * ws_dev off the mean curve are considered faulty.
            Defaults to 2.0, which is two standard deviations meaning
            about 5% of the values will be marked faulty.
            pow_min (float, optional): Lower bound on the power
            production above which data should be filtered. Defaults to
            20.0.
            pow_max ([type], optional): Upper bound on the power
            production below which data should be filtered. If none is
            specified, will derive this value based on the estimated rated
            power production. Defaults to None.
        """
        print("Filtering data by WS std. dev...")

        # Default properties: must be arrays with length equal to n.o. turbines
        if pow_max is None:
            # Derive maximum power as 0.95 times the rated power of every turbine
            pow_max = 0.95 * np.array(self.est_rated_pow)

        # Format input variables as arrays of length num_turbs
        if isinstance(pow_min, (int, float)):
            pow_min = np.repeat(pow_min, self.nturbs_all)
        if isinstance(pow_max, (int, float)):
            pow_max = np.repeat(pow_max, self.nturbs_all)
        if isinstance(pow_bin_width, (int, float)):
            pow_bin_width = np.repeat(pow_bin_width, self.nturbs_all)
        if isinstance(max_ws_dev, (int, float)):
            max_ws_dev = np.repeat(max_ws_dev, self.nturbs_all)

        df = self.df
        df_filters = self.df_filters
        for ti in self.turbine_list:
            # Extract appropriate subset from dataframe
            ids = df_filters[ti]["status"] == 1
            cols = ["ws_%03d" % ti, "pow_%03d" % ti]
            df_ok = df.loc[ids, cols].copy()

            # Filter by standard deviation for the subset
            if all(np.isnan(df_ok["pow_%03d" % ti].astype(float))):
                out_of_dev_series = [False] * df.shape[0]
            else:
                out_of_dev_series = filters.bin_filter(
                    bin_col=df_ok["pow_%03d" % ti].astype(float),
                    value_col=df_ok["ws_%03d" % ti].astype(float),
                    bin_width=pow_bin_width[ti],
                    threshold=max_ws_dev[ti],
                    center_type="median",
                    bin_min=pow_min[ti],
                    bin_max=pow_max[ti],
                    threshold_type="scalar",
                    direction="all",
                )

            # Save found outliers to array
            out_of_dev_indices = df_ok.index[np.where(out_of_dev_series)[0]]
            df_out_of_ws_dev = np.zeros(self.df.shape[0])
            df_out_of_ws_dev[out_of_dev_indices] = 1
            self.df_filters[ti]["ws_std_dev_outlier"] = [
                bool(i) for i in df_out_of_ws_dev
            ]
            print(
                "Removed %d outliers using WS standard deviation filtering."
                % (int(sum(df_out_of_ws_dev)))
            )

        # Finally, update status columns in dataframe
        self._update_status_flags()

    def save_df(self, fout):
        """Apply all filters to the dataframe by marking any fauilty data
        as None/np.nan. Then, save the dataframe to the specified path.

        Args:
            fout ([str]): Destination path for the output .ftr file.

        Returns:
            df ([pd.DataFrame]): Processed dataframe.
        """
        if not (self.turbine_list == self.full_turbs_list):
            print(
                "Skipping saving dataframe since not all turbines are filtered."
            )
            print(
                "Please specify 'turbine_list' as 'full' and filter accordingly before saving."
            )
            return None

        df = self.df.copy()
        for ti in self.turbine_list:
            bad_ids = self.df_filters[ti]["status"] == 0
            df = dff.df_mark_turbdata_as_faulty(
                df=df, cond=bad_ids, turbine_list=ti, verbose=True
            )

        # Reset index and save to file
        if "time" in df.columns:
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index(drop=False)

        df.to_feather(fout)
        return df

    def save_power_curve(self, fout="power_curve.csv"):
        """Save the estimated power curve as a .csv to a prespecified path.
        """
        return self.pw_curve_df.to_csv(fout)

    def plot_power_curves(self):
        """Plot all turbines' power curves in a single figure. Also estimate
        and plot a mean turbine power curve.
        """
        fig, ax = plt.subplots()
        x = np.array(self.pw_curve_df["ws"], dtype=float)
        for ti in self.turbine_list:
            ax.plot(x, self.pw_curve_df["pow_%03d" % ti], color="lightgray")

        pow_cols = ["pow_%03d" % ti for ti in self.turbine_list]
        pow_mean_array = self.pw_curve_df[pow_cols].mean(axis=1)
        pow_std_array = self.pw_curve_df[pow_cols].std(axis=1)

        yl = np.array(pow_mean_array - 2 * pow_std_array)
        yu = np.array(pow_mean_array + 2 * pow_std_array)
        ax.fill_between(
            np.hstack([x, x[::-1]]),
            np.hstack([yl, yu[::-1]]),
            color="tab:red",
            label="Uncertainty bounds (2 std. dev.)",
            alpha=0.30,
        )
        ax.plot(x, pow_mean_array, color="tab:red", label="Mean curve")
        ax.legend()

        return fig, ax

    def plot(
        self,
        draw_windows=True,
        confirm_plot=False,
        fi=None,
        save_path=None,
        fig_format="png",
        dpi=300,
    ):
        """Plot the wind speed power curve and mark faulty data according to
        their filters.

        Args:
            draw_windows (bool, optional): Plot the windows over which data
            is filtered. Defaults to True.
            confirm_plot (bool, optional): Add a secondary subplot showing
            which data are faulty and which are fine. Useful for debugging.
            Defaults to False.
            fi ([type], optional): floris object. If specified, will use
            this to plot the turbine power curves as implemented in floris.
            Defaults to None.
            save_path ([str], optional): Path to save the figure to. If none
            is specified, then will not save any figures. Defaults to None.
            fig_format (str, optional): Figure format if saved. Defaults to
            "png".
            dpi (int, optional): Image resolution if saved. Defaults to 300.
        """
        df = self.df

        fig_list = []
        for ti in self.turbine_list:
            print("Generating ws-power plot for turbine %03d" % ti)
            if confirm_plot:
                fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax = [ax]

            fig_list.append(fig)

            # Get filter dataframe
            df_f = self.df_filters[ti]
            N = df_f.shape[0]

            # Show the acceptable points
            ax[0].plot(
                df.loc[df_f["status"] == 1, "ws_%03d" % ti],
                df.loc[df_f["status"] == 1, "pow_%03d" % ti],
                ".",
                color="k",
                markersize=5,
                alpha=0.15,
                rasterized=True,
                label="Useful (%.1f %%)" % (100.0 * df_f["status"].sum() / N),
            )

            cols = [c for c in df_f.columns if "status" not in c]
            for c in cols:
                if df_f[c].sum() > 0:
                    # Show the points from filter 'c'
                    ax[0].plot(
                        df.loc[df_f[c], "ws_%03d" % ti],
                        df.loc[df_f[c], "pow_%03d" % ti],
                        ".",
                        markersize=5,
                        alpha=0.15,
                        rasterized=True,
                        label="%s (%.1f %%)" % (c, 100.0 * df_f[c].sum() / N),
                    )

            # Show the approximated power curve, bounds and FLORIS curve, if applicable
            ax[0].plot(
                self.pw_curve_df["ws"],
                self.pw_curve_df["pow_%03d" % ti],
                "--",
                label="Approximate power curve",
            )
            if fi is not None:
                fi_turb = fi.floris.farm.turbines[ti]
                Ad = 0.25 * np.pi * fi_turb.rotor_diameter ** 2.0
                ws_array = np.array(fi_turb.power_thrust_table["wind_speed"])
                cp_array = np.array(fi_turb.fCpInterp(ws_array))
                rho = fi.floris.farm.air_density
                pow_array = (
                    0.5 * rho * ws_array ** 3.0 * Ad * cp_array * 1.0e-3
                )
                ax[0].plot(ws_array, pow_array, "--", label="FLORIS curve")

            if self.pw_curve_df_bounds is not None:
                ax[0].plot(
                    self.pw_curve_df_bounds["ws"],
                    self.pw_curve_df_bounds["pow_%03d_lb" % ti],
                    "--",
                    label="Left bound for power curve",
                )
                ax[0].plot(
                    self.pw_curve_df_bounds["ws"],
                    self.pw_curve_df_bounds["pow_%03d_rb" % ti],
                    "--",
                    label="Right bound for power curve",
                )

            if draw_windows:
                xlim = (0.0, 30.0)  # ax[0].get_xlim()
                ylim = ax[0].get_ylim()

                window_list = [
                    w for w in self.window_list if ti in w["turbines"]
                ]
                for window in window_list:
                    ws_range = window["ws_range"]
                    pow_range = window["pow_range"]
                    axis = window["axis"]
                    idx = window["idx"]

                    if axis == 0:
                        # Filtered region left of curve
                        ut.plot_redzone(
                            ax[0],
                            xlim[0],
                            pow_range[0],
                            ws_range[0] - xlim[0],
                            pow_range[1] - pow_range[0],
                            "%d" % idx,
                            ii=idx,
                        )
                        # Filtered region right of curve
                        ut.plot_redzone(
                            ax[0],
                            ws_range[1],
                            pow_range[0],
                            xlim[1] - ws_range[1],
                            pow_range[1] - pow_range[0],
                            "%d" % idx,
                            ii=idx,
                        )
                    else:
                        # Filtered region above curve
                        ut.plot_redzone(
                            ax[0],
                            ws_range[0],
                            pow_range[1],
                            ws_range[1] - ws_range[0],
                            ylim[1] - pow_range[1],
                            "%d" % idx,
                            ii=idx,
                        )
                        # Filtered region below curve
                        ut.plot_redzone(
                            ax[0],
                            ws_range[0],
                            ylim[0],
                            ws_range[1] - ws_range[0],
                            pow_range[0] - ylim[0],
                            "%d" % idx,
                            ii=idx,
                        )
                    # ax[0].add_patch(rect)

                ax[0].set_xlim(xlim)
                ax[0].set_ylim(ylim)

            ax[0].set_title("Turbine %03d" % ti)
            ax[0].set_ylabel("Power (kW)")
            ax[0].set_xlabel("Wind speed (m/s)")
            lgd = ax[0].legend()
            for l in lgd.legendHandles:
                # Force alpha in legend to 1.0
                l._legmarker.set_alpha(1)

            if confirm_plot:
                ut._make_confirmation_plot(df, self.df_filters, ti=ti, ax=ax[1])
                ax[1].set_ylabel("")

            fig.tight_layout()
            if save_path is not None:
                plt.savefig(
                    save_path + "/wspowcurve_%03d." % ti + fig_format, dpi=dpi
                )

        return fig_list

    def plot_outliers_vs_time(
        self, save_path=None, fig_format="png", dpi=300
    ):
        """Generate bar plot where each week of data is gathered and its
        filtering results will be shown relative to the data size of each
        week. This plot can particularly be useful to investigate whether
        certain weeks/time periods show a particular high number of faulty
        measurements. This can often be correlated with maintenance time
        windows and the user may opt to completely remove any measurements
        in the found time period from the dataset.

        Args:
            save_path ([str], optional): Path to save the figure to. If none
            is specified, then will not save any figures. Defaults to None.
            fig_format (str, optional): Figure format if saved. Defaults to
            "png".
            dpi (int, optional): Image resolution if saved. Defaults to 300.
        """
        df = self.df

        fig_list = []
        for ti in self.turbine_list:
            print("Producing time-outliers bar plot for turbine %03d." % ti)
            df_f = self.df_filters[ti]
            cols = [c for c in df_f.columns if "status" not in c]
            conds = list(np.array(df_f[cols], dtype=bool).T)
            fig, ax = dff.plot_highlight_data_by_conds(df, conds, ti)
            ax.legend(["All data"] + cols)
            fig_list.append(fig)

            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                fp = os.path.join(
                    save_path,
                    ("fault_timestamps_histogram_%03d." % ti) + fig_format,
                )
                plt.savefig(fp, dpi=dpi)

        return fig_list

    def apply_filtering_to_other_df(self, df_target, fout=None):
        """This function enables the user to implement the changes made in
        the dataframe at hand in other dataframes. For example, if the data
        in the dataframe self.df is sampled at 60 s, one may want to apply
        the changes made back to the original 1 s dataset.

        Args:
            df_target ([pd.DataFrame]): Targetted dataframe to apply the
            changes to.
            fout ([str], optional): Output path for the formatted Dataframe.
            Defaults to None.

        Returns:
            df_target ([pd.DataFrame]): Formatted dataframe.
        """
        turbines_in_target_df = [
            ti
            for ti in self.full_turbs_list
            if "pow_%03d" % ti in df_target.columns
        ]

        if df_target.shape[0] <= 2:
            print("Dataframe is too small to estimate dt from. Skipping...")
            if fout is not None:
                print("Saving dataframe to ", fout)
                df_target = df_target.reset_index(
                    drop=("time" in df_target.columns)
                )
                df_target.to_feather(fout)
            return df_target

        time_array_target = df_target["time"].values
        dt_target = fsut.estimate_dt(time_array_target)
        if dt_target >= self.dt:
            raise UserWarning(
                "This function only works with higher "
                + "resolution data. If you want to map this"
                + "to lower resolution data, simply use the"
                + "dataframe downsampling function."
            )

        for ti in turbines_in_target_df:
            print(
                "Applying filtering to target_df with dt = %.1f s, turbine %03d."
                % (dt_target.seconds, ti)
            )
            status_bad = self.df[("status_%03d" % ti)] == 0
            time_array_src_bad = self.df.loc[status_bad, "time"].values
            time_array_src_bad = pd.to_datetime(time_array_src_bad)
            stws = [[t - self.dt, t] for t in time_array_src_bad]
            bad_ids = fsato.find_window_in_time_array(
                time_array_src=time_array_target, seek_time_windows=stws
            )

            if bad_ids is not None:
                bad_ids = np.concatenate(bad_ids)
                print(
                    "  Marking entries as faulty in higher resolution dataframe..."
                )
                df_target = dff.df_mark_turbdata_as_faulty(
                    df=df_target, cond=bad_ids, turbine_list=ti, verbose=True
                )

        if fout is not None:
            print("Saving dataframe to ", fout)
            df_target = df_target.reset_index(
                drop=("time" in df_target.columns)
            )
            df_target.to_feather(fout)

        return df_target
