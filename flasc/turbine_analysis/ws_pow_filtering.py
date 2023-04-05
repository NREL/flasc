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

from ..turbine_analysis.find_sensor_faults import find_sensor_stuck_faults
from .. import utilities as flascutils
from ..dataframe_operations import dataframe_filtering as dff



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
    def _get_all_unique_flags(self):
        """Private function that grabs all the unique filter flags
        that are available in self.df_filters and returns them
        as a list of strings. This is helpful when plotting the
        various filter sources in a scatter plot, for example.

        Returns:
            all_flags (list): List with all unique flags available
                in self.df_filters, each entry being a string.
        """
        # Get a list of all flags and then get colors correspondingly
        all_flags = list(np.sort(np.unique(self.df_filters)))
        if "clean" in all_flags:
            # Sort to make sure "clean" is the first entry in the legend
            all_flags.remove("clean")
            all_flags = ["clean"] + all_flags

        return all_flags

    def _get_mean_power_curves(self, ws_bins=np.arange(0.0, 25.5, 0.5), df=None):
        """Calculates the mean power production in bins of the wind speed,
        for all turbines in the wind farm.

        Args:
            ws_bins ([iteratible], optional): Wind speed bins. Defaults to
                np.arange(0.0, 25.5, 0.5).
            df ([pd.DataFrame]): Dataframe containing the turbine data,
                formatted in the generic SCADA data format. Namely, the
                dataframe should at the very least contain the columns:
                  * Time of each measurement: time
                  * Wind speed of each turbine: ws_000, ws_001, ... 
                  * Power production of each turbine: pow_000, pow_001, ...

        Returns:
            pw_curve_df ([pd.DataFrame]): Dataframe containing the wind
                speed bins and the mean power production value for every
                turbine.
        """

        # If df unspecified, use the locally filtered variable
        if df is None:
            df = self.df

        # Create a dataframe to contain the averaged power curves
        ws_max = np.max(ws_bins)
        ws_min = np.min(ws_bins)
        pw_curve_df = pd.DataFrame(
            {
                "ws": (ws_bins[1::] + ws_bins[0:-1]) / 2,
                "ws_min": ws_bins[0:-1],
                "ws_max": ws_bins[1::],
            }
        )

        # Loop through every turbine
        for ti in range(self.n_turbines):
            # Extract the measurements and calculate the bin average
            ws = df["ws_%03d" % ti]
            pw = df["pow_%03d" % ti]
            bin_ids = (ws > ws_min) & (ws < ws_max)
            ws_clean = ws[bin_ids]
            pw_clean = pw[bin_ids]

            bin_array = np.searchsorted(ws_bins, ws_clean, side="left")
            bin_array = bin_array - 1  # 0 -> 1st bin, rather than before bin
            pow_bins = [
                np.median(pw_clean[bin_array == i])
                for i in range(pw_curve_df.shape[0])
            ]

            # Write outputs to the dataframe
            pw_curve_df["pow_%03d" % ti] = pow_bins

        # Save the finalized power curve to self and return it to the user
        self.pw_curve_df = pw_curve_df
        return pw_curve_df

    def _reset_df(self):
        """Reset the 'filtered' dataframe, self.df, to its original form,
        before any measurements were marked as faulty. """

        # Copy the original dataframe from self
        df = self._df_initial  
        self.df = df.reset_index(drop=("time" in df.columns))

        # Derive the total number of turbines in the dataframe
        self.n_turbines = flascutils.get_num_turbines(df)

        # Get mean power curve from data to start with
        self._get_mean_power_curves()

    # Public methods
    def reset_filters(self):
        """Reset all filter variables and assume all data is clean."""
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
        """This is a generic method to filter the dataframe for any particular
        condition, for a specific turbine or specific set of turbines. This
        provides a platform for user-specific queries to filter and then inspect
        the data with. You can call this function multiple times and the filters
        will aggregate chronologically. This filter directly cuts down the
        dataframe self.df to a filtered subset.

        A correct usage is, for example:
            ws_pow_filtering.filter_by_condition(
                condition=(ws_pow_filtering.df["pow_{:03d}".format(ti)] < -1.0e-6),
                label="Power below zero",
                ti=ti,
                verbose=True,
            )

        and:
            ws_pow_filtering.filter_by_condition(
                condition=(ws_pow_filtering.df["is_operation_normal_{:03d}".format(ti)] == False),
                label="Self-flagged (is_operation_normal==False)",
                ti=ti,
                verbose=True,
            )

        Args:
            condition (iteratible): List or array-like variable with bool entries
                depicting whether the condition is met or not. These should be
                situations in which you classify the data as faulty. For example,
                high wind speeds but low power productions, or NaNs, self-flagged
                status variables.
            label (str): Name or description of the fault/condition that is flagged.
            ti (int): Turbine indentifier, typically an integer, but may also be a
                list. This flags the measurements of all these turbines as faulty
                for which condition==True.
            verbose (bool, optional): Print information to console. Defaults to True.
            apply_filters_to_df (bool, optional): Assign the flagged measurements in 
                self.df directly as NaN. Defaults to True.

        Returns:
            df_out: The filtered dataframe. All measurements that are flagged as faulty
                are overwritten by "None"/"NaN". If apply_filters_to_df==True, then this
                dataframe is equal to the internally filtered dataframe 'self.df'.
        """

        # Pour it into a list format
        if isinstance(ti, int):
            ti = [ti]

        # Load the dataframe from self
        df_in = self.df

        # Create standalone copy that we can manipulate, if apply_filters_to_df==False
        if not apply_filters_to_df:
            df_in = df_in.copy()

        # Mark data as faulty on the dataframe
        N_pre = [dff.df_get_no_faulty_measurements(df_in, tii) for tii in ti]
        df_out = dff.df_mark_turbdata_as_faulty(df=df_in, cond=condition, turbine_list=ti)

        # Print the reduction in useful data to the console, if verbose
        if verbose:
            for iii, tii in enumerate(ti):
                N_post = dff.df_get_no_faulty_measurements(df_out, tii)
                print(
                    "Faulty measurements for WTG {:03d} increased from {:.3f} % to {:.3f} %. Reason: '{:s}'.".format(
                        tii, 100.0 * N_pre[iii] / df_in.shape[0], 100.0 * N_post / df_in.shape[0], label
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
        """Filter the turbine measurements for sensor-stuck type of faults. This is
        the situation where a turbine measurement reads the exact same value for
        multiple consecutive timestamps. This typically indicates a "frozen" sensor
        rather than a true physical effect. This is particularly the case for
        signals that are known to change at a high rate and are measured with high
        precision, e.g., wind speed and wind direction measurements.

        Args:
            columns (list): List of columns which should be checked for sensor-stuck
                type of faults. A typical choice is ["ws_000", "wd_000"] with ti=0,
                which are the wind speed and wind direction for turbine 0. We can
                safely assume that those measurements should change between every
                10-minute measurement. Note that you may not want to include "pow_000",
                since that measurement may be constant for longer periods of time even
                during normal operation, e.g., when the turbine is shutdown at very
                low wind speeds or when the turbine is operating above rated wind
                speed. Note that if any of the signals in 'columns' is flagged as
                frozen ("stuck"), all measurements of that turbine will be marked 
                faulty.
            ti (int): The turbine identifier for which its measurements should be
                flagged as faulty when the signals in the columns are found to be
                frozen ("stuck"). This is typically the turbine number that corresponds
                to the columns, e.g., if you use  columns=["ws_000", "wd_000"] then 
                ti=0, and if you use  ["ws_003", "wd_003"] you use ti=3.
            n_consecutive_measurements (int, optional): Number of consecutive
                measurements that should read the same value for the measurement to be
                considered "frozen". Defaults to 3.
            stddev_threshold (float, optional): Threshold value, typically a low number.
                If the set of consecutive measurements do not differ by more than this
                value, then the measurements is considered stuck. Defaults to 0.001.
            plot (bool, optional): Produce plots highlighting a handful of situations
                in which the measurements are stuck in time. This is typically only
                helpful if you have more than 1% of measurements being faulty, and
                you would like to figure out whether this is a numerical issue or
                this is actually happening. Defaults to False.
            verbose (bool, optional): Print information to console. Defaults to True.

        Returns:
            self.df: Pandas DataFrame with the filtered data, in which faulty turbine
                measurements are flagged as None/NaN. This is an aggregated filtering
                variable, so it includes faulty-flagged measurements from filter
                operations in previous steps.
        """

        # Filter sensor faults using the separate function call
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

        # Apply the actual filter to the dataset
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
        cutoff_ws=20.0,
    ):
        """Filter the data by offset from the mean power curve in x-
        directions. This is an iterative process because the estimated mean
        curve actually changes as data is filtered. This process typically
        converges within a couple iterations.

        Args:
            ti (int): The turbine identifier for which the data should be
            filtered.
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
            cutoff_ws (float, optional): Upper limit for the filtering to occur.
            Typically, this is a value just below the cut-out wind speed. Namely,
            issues arise if you put this wind speed above the cut-out wind speed,
            because we effectively end up with two curves for the same power
            production (one at region 2, one going down from cut-out wind speed).
            This confuses the algorithm. Hence, suggested to put this somewhere
            around 15-25 m/s. Defaults to 20 m/s.
        """
        # Initialize the dataframe from self, as a starting point. Note
        # that in each iteration, we do not want to build upon the
        # filtered dataset from the previous iteration, because that
        # erroneously removes too much data. Instead, we start with the
        # same dataset every iteration but apply a slightly different filter.
        # The filter differs because the data the classify as faulty (based on
        # the estimated power curve) changes every iteration, and hence so
        # do the estimated mean power curves again. This explains the
        # iterative nature of the problem.
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
        """Return the filtered dataframe to the user.

        Returns:
            self.df: Pandas DataFrame with the filtered data, in which faulty turbine
                measurements are flagged as None/NaN. This is an aggregated filtering
                variable, so it includes faulty-flagged measurements from filter
                operations in previous steps.
        """
        return self.df

    def get_power_curve(self):
        """Return the turbine estimated mean power curves to the user.

        Returns:
            pw_curve_df ([pd.DataFrame]): Dataframe containing the wind
                speed bins and the mean power production value for every
                turbine.
        """
        return self.pw_curve_df

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
        """Plot the filtered data in a scatter plot, categorized
        by the source of their filter/fault. This is a generic
        function that allows the user to plot various numeric
        variables on the x and y axis.

        Args:
            ti (int): Turbine identifier. This is used to determine
                which turbine's filter history should be looked at.
            x_col (str): Column name to plot on the x-axis. A common
                choice is "ws_000" for ti=0, for example.
            y_col (str): Column name to plot on the y-axis. A common
                choice is "pow_000" for ti=0, for example.
            ax (plt.Axis, optional): Pyplot Figure axis in which the
                figure should be produced. If None specified, then
                 creates a new figure. Defaults to None.

        Returns:
            ax: The figure axis in which the scatter plot is drawn.
        """
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
            fi (FlorisInterface, optional): floris object. If not None, will
            use this to plot the turbine power curves as implemented in floris.
            Defaults to None.
            ax (plt.Axis): Pyplot Axis object.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

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
            fi (FlorisInterface, optional): floris object. If not None, will
            use this to plot the turbine power curves as implemented in floris.
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
        if ax is None:
            _, ax = plt.subplots(figsize=(13, 7))

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
