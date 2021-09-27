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

from floris_scada_analysis import time_operations as fsato
from floris_scada_analysis import utilities as fsut
from floris_scada_analysis.dataframe_operations import \
    dataframe_filtering as dff
from floris_scada_analysis.turbine_analysis import \
    ws_pow_filtering_utilities as ut

from operational_analysis.toolkits import filters


class ws_pw_curve_filtering:
    def __init__(self, df, turbine_list="all",
                 add_default_windows=False, rated_powers=None):

        self._set_df(df)
        self._set_turbine_mode(turbine_list=turbine_list)

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
        self.df = df.reset_index(drop=("time" in df.columns))
        self.dt = fsut.estimate_dt(self.df["time"])
        self.nturbs_all = fsut.get_num_turbines(df)
        self.full_turbs_list = range(self.nturbs_all)

    def _set_turbine_mode(self, turbine_list):
        if isinstance(turbine_list, str):
            if turbine_list == "all":
                num_turbines = fsut.get_num_turbines(self.df)
                turbine_list = range(num_turbines)
            else:
                raise KeyError("Invalid turbine_list specified.")

        self.turbine_list = turbine_list
        self.num_turbines = len(turbine_list)

    def _add_default_windows(self):
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
                "Adding window[%d] and window[%d] for turbines:" % (idx, idx + 1),
                ut.convert_list_to_ranges(turbs),
            )
            self.window_add(default_w0_ws, default_w0_pw, axis=0, turbines=turbs)
            self.window_add(default_w1_ws, default_w1_pw, axis=1, turbines=turbs)

    def _get_mean_power_curve(self, ws_bins=np.arange(0.0, 25.5, 0.5)):
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
                np.median(pw_clean[bin_array == i]) for i in range(pw_curve_df.shape[0])
            ]

            # Write outputs to the dataframe
            pw_curve_df["pow_%03d" % ti] = pow_bins
            self.pw_curve_df = pw_curve_df

        return pw_curve_df

    def _update_status_flags(self, verbose=True):
        for df_f in self.df_filters:
            cols = [c for c in df_f.columns if 'status' not in c]
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

        self._get_mean_power_curve()  # Update mean power curve

    # Public methods
    def reset_filters(self):
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
                    "mean_pow_curve_outlier": [False] * self.df.shape[0]
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
        if not isinstance(ids_to_remove, (list, np.array)):
            ids_to_remove = [ids_to_remove]
        ids_to_remove = np.sort(ids_to_remove)[::-1]
        for i in ids_to_remove:
            self.window_list.pop(i)

        # Update indices
        for i in range(len(self.window_list)):
            self.window_list[i]["idx"] = i

    def window_remove_all(self):
        self.window_list = []

    def window_print_all(self):
        for i in range(len(self.window_list)):
            window = self.window_list[i]
            for k in window.keys():
                if k == "turbines":
                    str_short = ut.convert_list_to_ranges(self.window_list[i][k])
                    print("window_list[%d][%s] = " % (i, k), str_short)
                elif not k == "idx":
                    print("window_list[%d][%s] = " % (i, k), self.window_list[i][k])
            print("")

    def filter_by_windows(self):
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
            self.df_filters[ti]["window_outlier"] = [bool(i) for i in df_out_of_windows]

        # Finally, update status columns in dataframe
        self._update_status_flags()

    def filter_by_power_curve(
        self, m_ws_lb=0.95, m_pow_lb=1.01, m_ws_rb=1.05, m_pow_rb=0.99, no_iterations=10
    ):
        print("Filtering data by deviations from the mean power curve...")
        for ii in range(no_iterations):
            # Create upper and lower bounds around mean curve
            df_xy = self.pw_curve_df.copy()
            x = np.array(df_xy["ws"], dtype=float)
            self.pw_curve_df_bounds = pd.DataFrame({"ws": x})
            for ti in self.turbine_list:
                y = np.array(df_xy["pow_%03d" % ti], dtype=float)

                # Create interpolants to left and right of mean curve
                ws_array = np.array(self.df["ws_%03d" % ti], dtype=float)
                pow_array = np.array(self.df["pow_%03d" % ti], dtype=float)
                ws_lb = np.interp(
                    x=pow_array,
                    xp=y * m_pow_lb,
                    fp=x * m_ws_lb,
                    left=np.nan,
                    right=np.nan,
                )
                ws_rb = np.interp(
                    x=pow_array,
                    xp=y * m_pow_rb,
                    fp=x * m_ws_rb,
                    left=np.nan,
                    right=np.nan,
                )

                out_of_bounds = (ws_array < ws_lb) | (ws_array > ws_rb)
                self.df_filters[ti]["mean_pow_curve_outlier"] = out_of_bounds

                # Write left and right bound to own curve
                self.pw_curve_df_bounds["pow_%03d_lb" % ti] = np.interp(
                    x=x, xp=x * m_ws_lb, fp=y * m_pow_lb, left=np.nan, right=np.nan
                )
                self.pw_curve_df_bounds["pow_%03d_rb" % ti] = np.interp(
                    x=x, xp=x * m_ws_rb, fp=y * m_pow_rb, left=np.nan, right=np.nan
                )

            # Update status flags and re-estimate mean power curve
            verbose = (ii == no_iterations - 1)  # Only print final iteration
            self._update_status_flags(verbose=verbose)

    def filter_by_wsdev(
        self, pow_bin_width=20.0, max_ws_dev=2.0, pow_min=20.0, pow_max=None
    ):
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
        for ti in self.turbine_list:
            # Extract appropriate subset from dataframe
            ids = df["status_%03d" % ti] == 1
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
            self.df_filters[ti]["ws_std_dev_outlier"] = [bool(i) for i in df_out_of_ws_dev]
            print(
                "Removed %d outliers using WS standard deviation filtering."
                % (int(sum(df_out_of_ws_dev)))
            )

        # Finally, update status columns in dataframe
        self._update_status_flags()

    def save_df(self, fout):
        if not (self.turbine_list == self.full_turbs_list):
            print("Skipping saving dataframe since not all turbines are filtered.")
            print(
                "Please specify 'turbine_list' as 'full' and filter accordingly before saving."
            )
            return None

        df = self.df.copy()
        for ti in self.turbine_list:
            bad_ids = (self.df_filters[ti]["status"] == 0)
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
        return self.pw_curve_df.to_csv(fout)

    def plot(
        self,
        draw_windows=True,
        confirm_plot=False,
        save_path=None,
        fig_format="png",
        dpi=300,
    ):
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

            # Show the approximated power curve
            ax[0].plot(
                self.pw_curve_df["ws"],
                self.pw_curve_df["pow_%03d" % ti],
                "--",
                label="Approximate power curve",
            )
            if self.pw_curve_df_bounds is not None:
                ax[0].plot(
                    self.pw_curve_df_bounds["ws"],
                    self.pw_curve_df_bounds["pow_%03d_lb" % ti],
                    "--",
                    color="tab:red",
                    label="Left bound for power curve",
                )
                ax[0].plot(
                    self.pw_curve_df_bounds["ws"],
                    self.pw_curve_df_bounds["pow_%03d_rb" % ti],
                    "--",
                    color="tab:pink",
                    label="Right bound for power curve",
                )

            if draw_windows:
                xlim = (0.0, 30.0)  # ax[0].get_xlim()
                ylim = ax[0].get_ylim()

                window_list = [w for w in self.window_list if ti in w["turbines"]]
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
            ax[0].legend()

            if confirm_plot:
                ut._make_confirmation_plot(df, ti=ti, ax=ax[1])
                ax[1].set_ylabel("")

            fig.tight_layout()
            if save_path is not None:
                plt.savefig(save_path + "/wspowcurve_%03d." % ti + fig_format, dpi=dpi)

        return fig_list

    def plot_outliers_vs_time(self, save_path=None, fig_format="png", dpi=300):
        df = self.df

        fig_list = []
        for ti in self.turbine_list:
            print("Producing time-outliers bar plot for turbine %03d." % ti)
            df_f = self.df_filters[ti]
            cols = [c for c in df_f.columns if 'status' not in c]
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
        turbines_in_target_df = [
            ti for ti in self.full_turbs_list if "pow_%03d" % ti in df_target.columns
        ]

        if df_target.shape[0] <= 2:
            print("Dataframe is too small to estimate dt from. Skipping...")
            if fout is not None:
                print("Saving dataframe to ", fout)
                df_target = df_target.reset_index(drop=("time" in df_target.columns))
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
                print("  Marking entries as faulty in higher resolution dataframe...")
                df_target = dff.df_mark_turbdata_as_faulty(
                    df=df_target, cond=bad_ids, turbine_list=ti, verbose=True
                )

        if fout is not None:
            print("Saving dataframe to ", fout)
            df_target = df_target.reset_index(drop=("time" in df_target.columns))
            df_target.to_feather(fout)

        return df_target
