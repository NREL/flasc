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

# from operational_analysis.toolkits import filters

from ..turbine_analysis.find_sensor_faults import find_sensor_stuck_faults
from .. import time_operations as fsato, utilities as fsut
from ..dataframe_operations import dataframe_filtering as dff
# from ..turbine_analysis import ws_pow_filtering_utilities as ut



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
    def __init__(self, df):
        """Initializes the class.

        Args:
            df ([pd.DataFrame]): Dataframe containing the turbine data,
                formatted in the generic SCADA data format. Namely, the
                dataframe should at the very least contain the columns:
                  * Time of each measurement: time
                  * Wind speed of each turbine: ws_000, ws_001, ... 
                  * Power production of each turbine: pow_000, pow_001, ...
        """

        # Write dataframe to self
        self._df_initial = df.copy()
        self.reset_filters()

    # Private methods
    def _reset_df(self):
        """Format and save the provided dataframe to the class.

        Args:
            df ([pd.DataFrame]): Dataframe containing the turbine data,
                formatted in the generic SCADA data format. Namely, the
                dataframe should at the very least contain the columns:
                  * Time of each measurement: time
                  * Wind speed of each turbine: ws_000, ws_001, ... 
                  * Power production of each turbine: pow_000, pow_001, ...
        """
        
        df = self._df_initial  # Copy the original dataframe from self
        self.df = df.reset_index(drop=("time" in df.columns))
        self.dt = fsut.estimate_dt(self.df["time"])

        # Get number of turbines in the dataframe
        self.n_turbines = fsut.get_num_turbines(df)

        # Get mean power curve to start with
        self._get_mean_power_curves()

    def _get_mean_power_curves(self, ws_bins=np.arange(0.0, 25.5, 0.5), df=None):
        """Calculates the mean power production in bins of the wind speed.

        Args:
            ws_bins ([iteratible], optional): Wind speed bins. Defaults to
                np.arange(0.0, 25.5, 0.5).

        Returns:
            pw_curve_df ([pd.DataFrame]): Dataframe containing the wind
                speed bins and the mean power production value for every
                turbine.
        """

        # If df unspecified, use the locally filtered variable
        if df is None:
            df = self.df

        ws_max = np.max(ws_bins)
        ws_min = np.min(ws_bins)
        pw_curve_df = pd.DataFrame(
            {
                "ws": (ws_bins[1::] + ws_bins[0:-1]) / 2,
                "ws_min": ws_bins[0:-1],
                "ws_max": ws_bins[1::],
            }
        )

        for ti in range(self.n_turbines):
            ws = df["ws_%03d" % ti]
            pw = df["pow_%03d" % ti]
            clean_ids = (ws > ws_min) & (ws < ws_max)
            ws_clean = ws[clean_ids]
            pw_clean = pw[clean_ids]

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

    # Public methods
    def reset_filters(self):
        """Reset all filter variables and assume all data is clean."""
        # Reset certain variables
        self.pw_curve_df_bounds = None

        # Reset the filtered dataframe to the original, unfiltered one
        self._reset_df()

        # Reset the dataframe with filter flags to mark all data as clean, initially
        all_clean_array = [
            ["clean" for _ in range(self.n_turbines)]
            for _ in range(self.df.shape[0])
        ]
        self.df_filters = pd.DataFrame(
            all_clean_array,
            index=self.df.index,
            columns=["WTG_{:03d}".format(ti) for ti in range(self.n_turbines)]
        )

    def filter_by_condition(
        self,
        condition,
        label,
        ti: int,
        verbose: bool = True,
        apply_filters_to_df: bool = True,
    ):

        # Pour it into a list format
        if isinstance(ti, int):
            ti = [ti]

        # Load the dataframe from self
        df_in = self.df

        # Create standalone copy that we can manipulate, if apply_filters_to_df==False
        if not apply_filters_to_df:
            df_in = df_in.copy()

        # Mark data as faulty on the dataframe
        df_out = dff.df_mark_turbdata_as_faulty(df=df_in, cond=condition, turbine_list=ti)

        if verbose:
            for tii in ti:
                N_pre = dff.df_get_no_faulty_measurements(df_in, tii)
                N_post = dff.df_get_no_faulty_measurements(df_out, tii)
                print(
                    "Faulty measurements for WTG {:03d} increased from {:.3f} % to {:.3f} %. Reason: '{:s}'.".format(
                        tii, 100.0 * N_pre / df_in.shape[0], 100.0 * N_post / df_in.shape[0], label
                    )
                )

        if apply_filters_to_df:
            # Update dataframe and filter labels
            for tii in ti:  
                self.df_filters.loc[condition, "WTG_{:03d}".format(tii)] = label

            # Recalculate mean power curves
            self._get_mean_power_curves()

        return df_out

    def filter_by_sensor_stuck_faults(
            self,
            columns: list,
            ti: int,
            n_consecutive_measurements: int = 3,
            stddev_threshold: float = 0.001,
            plot: bool = False,
            verbose: bool = True
        ):
        # Filter sensor faults
        stuck_indices = find_sensor_stuck_faults(
            df=self.df,
            columns=columns,
            ti=ti,
            stddev_threshold=stddev_threshold,
            n_consecutive_measurements=n_consecutive_measurements,
            plot_figures=plot,
        )

        # Convert to a condition format
        flag_array = np.zeros(self.df.shape[0], dtype=bool)
        flag_array[stuck_indices] = True

        self.filter_by_condition(
            condition=flag_array,
            label="Sensor-stuck fault",
            ti=ti,
            verbose=verbose,
            apply_filters_to_df=True,
        )

        return self.df

    def filter_by_power_curve(
        self,
        ti,
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

        df_initial_filtered = self.df.copy()

        # Iteratively filter data and recalculate the mean power curves
        for ii in range(no_iterations):
            # Only print final iteration
            is_final_iteration = (ii == no_iterations - 1)

            # Create upper and lower bounds around mean curve
            df_xy = self.pw_curve_df.copy()
            x_full = np.array(df_xy["ws"], dtype=float)
            x = x_full[x_full < cutoff_ws]  # Only filter until 15 m/s
            self.pw_curve_df_bounds = pd.DataFrame({"ws": x})

            y = np.array(df_xy["pow_%03d" % ti], dtype=float)
            y = y[x_full < cutoff_ws]  # Only filter until 15 m/s
            if np.all(np.isnan(y)):
                self.pw_curve_df_bounds["pow_%03d_lb" % ti] = None
                self.pw_curve_df_bounds["pow_%03d_rb" % ti] = None
                continue

            # Create interpolants to left and right of mean curve
            ws_array = np.array(df_initial_filtered["ws_%03d" % ti], dtype=float)
            pow_array = np.array(df_initial_filtered["pow_%03d" % ti], dtype=float)

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

            # Filter the dataframe according to hypothetical power curve
            df_iteration = self.filter_by_condition(
                condition=(ws_array < ws_lb) | (ws_array > ws_rb),
                label="Mean power curve outlier",
                ti=ti,
                verbose=is_final_iteration,  # If final iteration, be verbose
                apply_filters_to_df=is_final_iteration,  # If final iteration, save dataframe to self
            )

            # Recalculate the mean power curve based on current iteration's filtered dataframe
            self._get_mean_power_curves(df=df_iteration)
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

        return self.df

    def filter_by_floris_power_curve(
        self,
        fi,
        ti,
        m_ws_lb=0.95,
        m_pow_lb=1.01,
        m_ws_rb=1.05,
        m_pow_rb=0.99,
        ws_deadband=0.50,
        pow_deadband=20.0,
        cutoff_ws=25.0,
    ):
        """Filter the data by offset from the floris power curve in x-
        directions.

        Args:
            fi (FlorisInterface): The FlorisInterface object for the farm
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
        """
        print("Filtering data by deviations from the floris power curve...")

        # Create upper and lower bounds around floris curve
        df_xy = self.pw_curve_df.copy()
        rho = fi.floris.flow_field.air_density
        for ti in range(len(fi.layout_x)):
            fi_turb = fi.floris.farm.turbine_definitions[ti]
            Ad = 0.25 * np.pi * fi_turb["rotor_diameter"] ** 2.0
            ws_array = np.array(fi_turb["power_thrust_table"]["wind_speed"])
            cp_array = np.array(fi_turb["power_thrust_table"]["power"])
            pow_array = (
                0.5 * rho * ws_array ** 3.0 * Ad * cp_array * 1.0e-3
            )
            df_xy.loc[df_xy.index, "pow_{:03d}".format(ti)] = (
                np.interp(xp=ws_array, fp=pow_array, x=df_xy["ws"])
            )

        x_full = np.array(df_xy["ws"], dtype=float)
        x = x_full[x_full < cutoff_ws]
        self.pw_curve_df_bounds = pd.DataFrame({"ws": x})

        y = np.array(df_xy["pow_%03d" % ti], dtype=float)
        y = y[x_full < cutoff_ws]
        if np.all(np.isnan(y)):
            self.pw_curve_df_bounds["pow_%03d_lb" % ti] = None
            self.pw_curve_df_bounds["pow_%03d_rb" % ti] = None
            return self.df  # Do nothing

        # Create interpolants to left and right of mean curve
        ws_array = np.array(df_iteration["ws_%03d" % ti], dtype=float)
        pow_array = np.array(df_iteration["pow_%03d" % ti], dtype=float)

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

        self.filter_by_condition(
            condition=(ws_array < ws_lb) | (ws_array > ws_rb),
            label="Outlier by FLORIS power curve",
            ti=ti,
            apply_filters_to_df=True,
        )

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

        return self.df

    def get_df(self):
        return self.df

    def save_df(self, fout):
        """Apply all filters to the dataframe by marking any fauilty data
        as None/np.nan. Then, save the dataframe to the specified path.

        Args:
            fout ([str]): Destination path for the output .ftr file.

        Returns:
            df ([pd.DataFrame]): Processed dataframe.
        """
        return self.df.to_feather(fout)

    def save_power_curve(self, fout="power_curve.csv"):
        """Save the estimated power curve as a .csv to a prespecified path.
        """
        return self.pw_curve_df.to_csv(fout)

    def plot_farm_mean_power_curve(self):
        """Plot all turbines' power curves in a single figure. Also estimate
        and plot a mean turbine power curve.
        """
        fig, ax = plt.subplots()
        x = np.array(self.pw_curve_df["ws"], dtype=float)
        for ti in range(self.n_turbines):
            ax.plot(x, self.pw_curve_df["pow_%03d" % ti], color="lightgray")

        pow_cols = ["pow_%03d" % ti for ti in range(self.n_turbines)]
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
        ax.set_title("Mean of all turbine power curves with UQ")

        return fig, ax

    def plot_filters_custom_scatter(self, ti, x_col, y_col, ax=None):
        # Create figure, if not specified
        if ax is None:
            _, ax = plt.subplots()

        # Get filter dataframe
        df_f = self.df_filters["WTG_{:03d}".format(ti)]
        all_flags = self._get_all_unique_flags()
        N = df_f.shape[0]

        # For each flagging condition, plot the results
        for flag in all_flags:
            ids = (df_f == flag)
            df_subset = self._df_initial.loc[ids]
            percentage = 100.0 * np.sum(ids) / N
            if any(ids):
                ax.plot(
                    df_subset[x_col],
                    df_subset[y_col],
                    ".",
                    markersize=5,
                    alpha=0.15,
                    rasterized=True,
                    label="{:s} ({:.2f} %)".format(flag, percentage),
                )

        lgd = ax.legend()
        for l in lgd.legendHandles:
            l.set_alpha(1)  # Force alpha in legend to 1.0

        ax.set_title("WTG {:03d}: Filters".format(ti))
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True)
    
        return ax

    def plot_filters_in_ws_power_curve(self, ti, fi=None, ax=None):
        """Plot the wind speed power curve and connect each faulty datapoint
        to the label it was classified as faulty with.

        Args:
            ti (int): Turbine number which should be plotted.
            fi ([type], optional): floris object. If specified, will use
            this to plot the turbine power curves as implemented in floris.
            Defaults to None.
            ax (plt.Axis): Pyplot Axis object.
        """

        if ax is None:
            _, ax = plt.subplots()

        # First use the custom filter plot to do the majority of the work
        self.plot_filters_custom_scatter(
            ti=ti,
            x_col="ws_{:03d}".format(ti),
            y_col="pow_{:03d}".format(ti),
            ax=ax,
        )

        # Show the approximated power curves, bounds and FLORIS curve, if applicable
        ax.plot(
            self.pw_curve_df["ws"],
            self.pw_curve_df["pow_%03d" % ti],
            "--",
            label="Approximate power curve",
        )

        if fi is not None:
            fi_turb = fi.floris.farm.turbine_definitions[ti]
            Ad = 0.25 * np.pi * fi_turb["rotor_diameter"] ** 2.0
            ws_array = np.array(fi_turb["power_thrust_table"]["wind_speed"])
            cp_array = np.array(fi_turb["power_thrust_table"]["power"])
            rho = fi.floris.flow_field.air_density
            pow_array = (
                0.5 * rho * ws_array ** 3.0 * Ad * cp_array * 1.0e-3
            )
            ax.plot(ws_array, pow_array, "--", label="FLORIS curve")

        if self.pw_curve_df_bounds is not None:
            ax.plot(
                self.pw_curve_df_bounds["ws"],
                self.pw_curve_df_bounds["pow_%03d_lb" % ti],
                "--",
                label="Left bound for power curve",
            )
            ax.plot(
                self.pw_curve_df_bounds["ws"],
                self.pw_curve_df_bounds["pow_%03d_rb" % ti],
                "--",
                label="Right bound for power curve",
            )

        lgd = ax.legend()
        for l in lgd.legendHandles:
            l.set_alpha(1)  # Force alpha in legend to 1.0

        ax.set_title("WTG {:03d}: Filters".format(ti))
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True)

        return ax

    def plot_postprocessed_in_ws_power_curve(self, ti, fi=None, ax=None):
        """Plot the wind speed power curve and mark faulty data according to
        their filters.

        Args:
            ti (int): Turbine number which should be plotted.
            fi ([type], optional): floris object. If specified, will use
            this to plot the turbine power curves as implemented in floris.
            Defaults to None.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        # Get filter dataframe
        df_f = self.df_filters["WTG_{:03d}".format(ti)]
        N = df_f.shape[0]

        # Plot the finalized dataframe results in a second subplot
        percentage = 100.0 * np.sum(df_f == "clean") / N
        ax.plot(
            self.df["ws_%03d" % ti],
            self.df["pow_%03d" % ti],
            ".",
            color="k",
            markersize=5,
            alpha=0.15,
            rasterized=True,
            label="Postprocessed dataset ({:.1f} %)".format(percentage),
        )

        # Show the approximated power curves, bounds and FLORIS curve, if applicable
        ax.plot(
            self.pw_curve_df["ws"],
            self.pw_curve_df["pow_%03d" % ti],
            "--",
            label="Approximate power curve",
        )

        if fi is not None:
            fi_turb = fi.floris.farm.turbine_definitions[ti]
            Ad = 0.25 * np.pi * fi_turb["rotor_diameter"] ** 2.0
            ws_array = np.array(fi_turb["power_thrust_table"]["wind_speed"])
            cp_array = np.array(fi_turb["power_thrust_table"]["power"])
            rho = fi.floris.flow_field.air_density
            pow_array = (
                0.5 * rho * ws_array ** 3.0 * Ad * cp_array * 1.0e-3
            )
            ax.plot(ws_array, pow_array, "--", label="FLORIS curve")

        if self.pw_curve_df_bounds is not None:
            ax.plot(
                self.pw_curve_df_bounds["ws"],
                self.pw_curve_df_bounds["pow_%03d_lb" % ti],
                "--",
                label="Left bound for power curve",
            )
            ax.plot(
                self.pw_curve_df_bounds["ws"],
                self.pw_curve_df_bounds["pow_%03d_rb" % ti],
                "--",
                label="Right bound for power curve",
            )

        lgd = ax.legend()
        for l in lgd.legendHandles:
            l.set_alpha(1)  # Force alpha in legend to 1.0

        ax.set_title("WTG {:03d}: Postprocessed dataset".format(ti))
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True)

        return ax

    def plot_filters_in_time(self, ti, ax=None):
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
        # Get a list of all flags and then get colors correspondingly
        all_flags = self._get_all_unique_flags()

        # Manipulate dataframe to easily plot results
        df_f = self.df_filters["WTG_{:03d}".format(ti)]
        df_conditional = pd.concat([pd.DataFrame({flag: np.array(df_f==flag, dtype=int)}) for flag in all_flags], axis=1)
        df_merged = pd.concat([df_conditional, self.df["time"]], axis=1)
        df_histogram = df_merged.groupby([df_merged["time"].dt.year, df_merged["time"].dt.isocalendar().week]).sum(numeric_only=True)

        # Plot the histogram information
        ax = df_histogram.plot.bar(stacked=True, ax=ax)
        ax.set_ylabel("Count (-)")
        ax.set_title("WTG {:03d}".format(ti))
        ax.grid(True)

        return ax

    def _get_all_unique_flags(self):
        # Get a list of all flags and then get colors correspondingly
        all_flags = list(np.sort(np.unique(self.df_filters)))
        if "clean" in all_flags:
            # Sort to make sure "clean" is the first entry in the legend
            all_flags.remove("clean")
            all_flags = ["clean"] + all_flags

        return all_flags
