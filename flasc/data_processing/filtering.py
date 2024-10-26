"""Implement filtering class and functions for FLASC data."""

import itertools

import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import Legend
from bokeh.palettes import Category20_20 as palette
from bokeh.plotting import ColumnDataSource, figure
from scipy.interpolate import interp1d

from flasc.data_processing import dataframe_manipulations as dfm
from flasc.data_processing.find_sensor_faults import find_sensor_stuck_faults
from flasc.logging_manager import LoggingManager
from flasc.utilities import utilities as flascutils

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def df_get_no_faulty_measurements(df, turbine):
    """Get the number of faulty measurements for a specific turbine.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe containing the turbine data,
            formatted in the generic SCADA data format. Namely, the
            dataframe should at the very least contain the columns:
              * Time of each measurement: time
              * Wind speed of each turbine: ws_000, ws_001, ...
              * Power production of each turbine: pow_000, pow_001, ...
        turbine (int): The turbine identifier for which the number of
            faulty measurements should be counted.

    Returns:
        N_isnan (int): Number of faulty measurements for the turbine.
    """
    if isinstance(turbine, str):
        turbine = int(turbine)
    entryisnan = np.isnan(df["pow_%03d" % turbine].astype(float))
    N_isnan = np.sum(entryisnan)
    return N_isnan


def df_mark_turbdata_as_faulty(df, cond, turbine_list, exclude_columns=[]):
    """Mark turbine data as faulty based on a condition.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe containing the turbine data,
            formatted in the generic SCADA data format.
        cond (iteratible): List or array-like variable with bool entries
            depicting whether the condition is met or not. These should be
            situations in which you classify the data as faulty. For example,
            high wind speeds but low power productions, or NaNs, self-flagged
            status variables.
        turbine_list (int, list): Turbine identifier(s) for which the data
            should be flagged as faulty when the condition is met.
        exclude_columns (list, optional): List of columns that should not
            be considered for the filtering. Defaults to [].

    Returns:
        pd.DataFrame | FlascDataFrame: Dataframe with the faulty measurements marked as None.
    """
    if isinstance(turbine_list, (np.integer, int)):
        turbine_list = [turbine_list]

    for ti in turbine_list:
        cols = [s for s in df.columns if s[-4::] == ("_%03d" % ti) and s not in exclude_columns]
        df.loc[cond, cols] = None  # Delete measurements

    return df


class FlascFilter:
    """Implement filtering class for SCADA data.

    This class allows a user to filter turbine data based on the
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

    def __init__(self, df, turbine_names=None):
        """Initializes the class.

        Args:
            df (pd.DataFrame | FlascDataFrame): Dataframe containing the turbine data,
                formatted in the generic SCADA data format. Namely, the
                dataframe should at the very least contain the columns:
                  * Time of each measurement: time
                  * Wind speed of each turbine: ws_000, ws_001, ...
                  * Power production of each turbine: pow_000, pow_001, ...
            turbine_names (list, optional): List of turbine names. Defaults to None.
        """
        # Write dataframe to self
        self._df_initial = df.copy()
        self.reset_filters()

        # Save the turbine names
        self.turbine_names = turbine_names

    # Private methods
    def _get_all_unique_flags(self):
        """Returns all unique flags in the dataframe.

        Private function that grabs all the unique filter flags
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

    def _reset_mean_power_curves(self, ws_bins=np.arange(0.0, 25.5, 0.5)):
        # If uninitialized, create an empty dataframe with NaNs
        pw_curve_dict = dict(
            zip(
                [f"pow_{ti:03d}" for ti in range(self.n_turbines)],
                [np.ones(len(ws_bins) - 1) * np.nan] * self.n_turbines,
            )
        )
        pw_curve_dict["ws"] = (ws_bins[1::] + ws_bins[0:-1]) / 2
        pw_curve_dict["ws_min"] = ws_bins[0:-1]
        pw_curve_dict["ws_max"] = ws_bins[1::]
        pw_curve_df = pd.DataFrame(pw_curve_dict)

        self._pw_curve_ws_bins = ws_bins
        self.pw_curve_df = pw_curve_df

    def _get_mean_power_curves(self, df=None, turbine_subset=None):
        """Calculates the mean power production in bins of the wind speed.

        Calculates the mean power production in bins of the wind speed,
        for all turbines in the wind farm.

        Args:
            ws_bins ([iteratible], optional): Wind speed bins. Defaults to
                np.arange(0.0, 25.5, 0.5).
            df (pd.DataFrame): Dataframe containing the turbine data,
                formatted in the generic SCADA data format. Namely, the
                dataframe should at the very least contain the columns:
                  * Time of each measurement: time
                  * Wind speed of each turbine: ws_000, ws_001, ...
                  * Power production of each turbine: pow_000, pow_001, ...
            turbine_subset (list, optional): List of turbine indices to
                calculate the mean power curve for. If None is specified,
                defaults to calculating it for all turbines.

        Returns:
            pd.DataFrame: Dataframe containing the wind
                speed bins and the mean power production value for every
                turbine.
        """
        # If df unspecified, use the locally filtered variable
        if df is None:
            df = self.df

        # Get existing power curve
        pw_curve_df = self.pw_curve_df

        # By default, if unspecified, Calculate power curve for all turbines
        if turbine_subset is None:
            turbine_subset = list(range(self.n_turbines))

        # Apply binning to the wind speeds of the turbine(s)
        ws_bin_cuts_subset = [
            pd.cut(df[f"ws_{ti:03d}"], bins=self._pw_curve_ws_bins) for ti in turbine_subset
        ]

        # Now add the binned wind speeds to the power measurements dataframe
        df_pow_and_ws_bins_subset = pd.concat(
            [df[["pow_%03d" % ti for ti in turbine_subset]], *ws_bin_cuts_subset], axis=1
        )

        # Now group power measurements by their wind speed bin and calculate the median
        pw_curve_df_subset = (
            pd.concat(
                [
                    df_pow_and_ws_bins_subset.groupby(by=f"ws_{ti:03d}", observed=False)[
                        f"pow_{ti:03d}"
                    ].median()
                    for ti in turbine_subset
                ],
                axis=1,
            )
            .sort_index()
            .reset_index(drop=True)
        )

        # Update the median power curve for the turbines in turbine_subset
        pw_curve_df[[f"pow_{ti:03d}" for ti in turbine_subset]] = pw_curve_df_subset

        # Save the finalized power curve to self and return it to the user
        self.pw_curve_df = pw_curve_df
        return pw_curve_df

    def _set_legend_alpha_to_one(self, lgd: matplotlib.legend.Legend) -> None:
        """Set the alpha value of the provided Legend object to be 1.

        Args:
            lgd (matplotlib.legend.Legend): Legend object
        """
        try:
            for h in lgd.legendHandles:
                h.set_alpha(1)  # Force alpha in legend to 1.0
        except AttributeError:
            # see https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.7.0.html#legend-legendhandles
            for h in lgd.legend_handles:
                h.set_alpha(1)

    # Public methods
    def reset_filters(self):
        """Reset all filter variables and assume all data is clean."""
        # Copy the original, unfiltered dataframe from self
        df = self._df_initial
        self.df = df.reset_index(drop=("time" in df.columns))
        self.n_turbines = flascutils.get_num_turbines(df)

        # Reset the dataframe with filter flags to mark all data as clean, initially
        all_clean_array = [
            ["clean" for _ in range(self.n_turbines)] for _ in range(self.df.shape[0])
        ]
        self.df_filters = pd.DataFrame(
            all_clean_array,
            index=self.df.index,
            columns=["WTG_{:03d}".format(ti) for ti in range(self.n_turbines)],
        )

        # Reset the mean power curves of the turbines
        self._reset_mean_power_curves()

    def filter_by_condition(
        self,
        condition,
        label,
        ti: int,
        verbose: bool = True,
        apply_filters_to_df: bool = True,
    ):
        """Filter the dataframe for a specific condition, for a specific turbine.

        This is a generic method to filter the dataframe for any particular
        condition, for a specific turbine or specific set of turbines. This
        provides a platform for user-specific queries to filter and then inspect
        the data with. You can call this function multiple times and the filters
        will aggregate chronologically. This filter directly cuts down the
        dataframe self.df to a filtered subset.

        A correct usage is, for example:
            FlascFilter.filter_by_condition(
                condition=(FlascFilter.df["pow_{:03d}".format(ti)] < -1.0e-6),
                label="Power below zero",
                ti=ti,
                verbose=True,
            )

        and:
            FlascFilter.filter_by_condition(
                condition=(FlascFilter.df["is_operation_normal_{:03d}".format(ti)] == False),
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
            ti (int): Turbine identifier, typically an integer, but may also be a
                list. This flags the measurements of all these turbines as faulty
                for which condition==True.
            verbose (bool, optional): Print information to console. Defaults to True.
            apply_filters_to_df (bool, optional): Assign the flagged measurements in
                self.df directly as NaN. Defaults to True.

        Returns:
            pd.Dataframe | FlascDataFrame: The filtered dataframe.
                All measurements that are flagged as faulty
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
        N_pre = [df_get_no_faulty_measurements(df_in, tii) for tii in ti]
        df_out = df_mark_turbdata_as_faulty(df=df_in, cond=condition, turbine_list=ti)

        # Print the reduction in useful data to the console, if verbose
        if verbose:
            for iii, tii in enumerate(ti):
                N_post = df_get_no_faulty_measurements(df_out, tii)
                logger.info(
                    (
                        "Faulty measurements for WTG {:03d} increased from {:.3f} % to {:.3f} "
                        "%. Reason: '{:s}'."
                    ).format(
                        tii,
                        100.0 * N_pre[iii] / df_in.shape[0],
                        100.0 * N_post / df_in.shape[0],
                        label,
                    )
                )

        if apply_filters_to_df:
            # Update dataframe and filter labels
            for tii in ti:
                self.df_filters.loc[condition, "WTG_{:03d}".format(tii)] = label

                # Clear the mean power curves. Namely, with this new filtering application
                # the mean power curves must be recalculated.
                self.pw_curve_df[f"pow_{tii:03d}"] = None  # Set as Nones

        return df_out

    def filter_by_sensor_stuck_faults(
        self,
        columns: list,
        ti: int,
        n_consecutive_measurements: int = 3,
        stddev_threshold: float = 0.001,
        plot: bool = False,
        verbose: bool = True,
    ):
        """Filter the turbine measurements for sensor-stuck type of faults.

        This is
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
            pd.Dataframe | FlascDataFrame: Pandas DataFrame with the filtered data,
                in which faulty turbine
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
        """Filter the data by offset from the mean power curve in x-directions.

        This is an iterative process because the estimated mean
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
            ws_deadband (float, optional): Deadband in [m/s] around the median
                power curve around which data is by default classified as valid.
                Defaults to 0.50.
            pow_deadband (float, optional): Deadband in [kW] around the median
                power curve around which data is by default classified as valid.
                Defaults to 20.0.
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
        df_initial_filtered = self.df[[f"ws_{ti:03d}", f"pow_{ti:03d}"]].copy()

        # Iteratively filter data and recalculate the mean power curves
        self._get_mean_power_curves(turbine_subset=[ti])
        for ii in range(no_iterations):
            # Only print final iteration
            is_final_iteration = ii == no_iterations - 1

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
            good_ids = np.hstack([(np.diff(lb_pow_l) >= 0.0), True]) & (
                ~np.isnan(lb_pow[0:id_center])
            )
            good_ids[0] = True
            lb_pow_l = np.interp(lb_ws_l, lb_ws_l[good_ids], lb_pow_l[good_ids])
            lb_pow[0:id_center] = lb_pow_l
            non_nans = ~np.isnan(lb_pow) & ~np.isnan(lb_ws)
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
            good_ids = np.hstack([(np.diff(rb_pow_l) >= 0.0), True]) & (
                ~np.isnan(rb_pow[0:id_center])
            )
            good_ids[0] = True
            rb_pow_l = np.interp(rb_ws_l, rb_ws_l[good_ids], rb_pow_l[good_ids])
            rb_pow[0:id_center] = rb_pow_l
            non_nans = ~np.isnan(rb_pow) & ~np.isnan(rb_ws)
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
                apply_filters_to_df=is_final_iteration,  # If final , save dataframe to self
            )

            # Recalculate the mean power curve based on current iteration's filtered dataframe
            self._get_mean_power_curves(df=df_iteration, turbine_subset=[ti])
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
        fm,
        ti,
        m_ws_lb=0.95,
        m_pow_lb=1.01,
        m_ws_rb=1.05,
        m_pow_rb=0.99,
        ws_deadband=0.50,
        pow_deadband=20.0,
        cutoff_ws=20.0,
    ):
        """Filter the data by offset from the floris power curve.

        Args:
            fm (FlorisModel): The FlorisModel object for the farm
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
            ws_deadband (float, optional): Deadband in [m/s] around the median
                power curve around which data is by default classified as valid.
                Defaults to 0.50.
            pow_deadband (float, optional): Deadband in [kW] around the median
                power curve around which data is by default classified as valid.
                Defaults to 20.0.
            cutoff_ws (float, optional): Wind speed up to which the median
                power curve is calculated and the data is filtered for. You should
                make sure this variable is set to a value above the rated wind
                speed and below the cut-out wind speed. If you are experiencing
                problems with data filtering and your data points have a downward
                trend near the high wind speeds, try decreasing this variable's
                value to 15.0.

        Returns:
            pd.Dataframe | FlascDataFrame: Pandas DataFrame with the filtered data,
                in which faulty turbine
                measurements are flagged as None/NaN. This is an aggregated filtering
                variable, so it includes faulty-flagged measurements from filter
                operations in previous steps.
        """
        logger.info("Filtering data by deviations from the floris power curve...")

        # Create upper and lower bounds around floris curve

        # Get mean power curves first, if not yet calculated
        if self.pw_curve_df[f"pow_{ti:03d}"].isna().all():
            self._get_mean_power_curves(turbine_subset=[ti])

        df_xy = self.pw_curve_df.copy()
        rho = fm.core.flow_field.air_density
        for ti in range(len(fm.layout_x)):
            fm_turb = fm.core.farm.turbine_definitions[ti]
            Ad = 0.25 * np.pi * fm_turb["rotor_diameter"] ** 2.0
            ws_array = np.array(fm_turb["power_thrust_table"]["wind_speed"])
            cp_array = np.array(fm_turb["power_thrust_table"]["power"])
            pow_array = 0.5 * rho * ws_array**3.0 * Ad * cp_array * 1.0e-3
            df_xy.loc[df_xy.index, "pow_{:03d}".format(ti)] = np.interp(
                xp=ws_array, fp=pow_array, x=df_xy["ws"]
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
        good_ids = np.hstack([(np.diff(lb_pow_l) >= 0.0), True]) & (~np.isnan(lb_pow[0:id_center]))
        good_ids[0] = True
        lb_pow_l = np.interp(lb_ws_l, lb_ws_l[good_ids], lb_pow_l[good_ids])
        lb_pow[0:id_center] = lb_pow_l
        non_nans = ~np.isnan(lb_pow) & ~np.isnan(lb_ws)
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
        good_ids = np.hstack([(np.diff(rb_pow_l) >= 0.0), True]) & (~np.isnan(rb_pow[0:id_center]))
        good_ids[0] = True
        rb_pow_l = np.interp(rb_ws_l, rb_ws_l[good_ids], rb_pow_l[good_ids])
        rb_pow[0:id_center] = rb_pow_l
        non_nans = ~np.isnan(rb_pow) & ~np.isnan(rb_ws)
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
        self._get_mean_power_curves(turbine_subset=[ti])  # Recalculate mean curve

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
           pd.DataFrame | FlascDataFrame: Pandas DataFrame with the filtered data,
                in which faulty turbine
                measurements are flagged as None/NaN. This is an aggregated filtering
                variable, so it includes faulty-flagged measurements from filter
                operations in previous steps.
        """
        return self.df

    def get_power_curve(self, calculate_missing=True):
        """Return the turbine estimated mean power curves to the user.

        Args:
            calculate_missing (bool, optional): Calculate the median power
                curves for the turbines for the turbines of which their
                power curves were previously not yet calculated.

        Returns:
            pd.DataFrame: Dataframe containing the estimated mean power curves.
        """
        if calculate_missing and (self.pw_curve_df.isna().all(axis=0).any()):
            turbine_subset = np.where(
                self.pw_curve_df[[f"pow_{ti:03d}" for ti in range(self.n_turbines)]]
                .isna()
                .all(axis=0)
            )[0]
            self._get_mean_power_curves(turbine_subset=turbine_subset)

        return self.pw_curve_df

    def plot_farm_mean_power_curve(self, fm=None):
        """Plot mean of all turbines' power curves and show individual curves.

        Also estimate
        and plot a mean turbine power curve.

        Args:
            fm (FlorisModel): The FlorisModel object for the farm. If
              specified by the user, then the farm-average turbine power curve
              from FLORIS will be plotted on top of the SCADA-based power curves.

        Returns:
            tuple (fig, ax): The figure and axis objects of the plot.

        """
        # Get mean power curves for the turbines that are not yet calculated
        if self.pw_curve_df.isna().all(axis=0).any():
            turbine_subset = np.where(
                self.pw_curve_df[[f"pow_{ti:03d}" for ti in range(self.n_turbines)]]
                .isna()
                .all(axis=0)
            )[0]
            self._get_mean_power_curves(turbine_subset=turbine_subset)

        # Create the figure
        fig, ax = plt.subplots()
        x = np.array(self.pw_curve_df["ws"], dtype=float)
        for ti in range(self.n_turbines):
            ax.plot(x, self.pw_curve_df["pow_%03d" % ti], color="lightgray")

        pow_cols = ["pow_%03d" % ti for ti in range(self.n_turbines)]
        pow_mean_array = self.pw_curve_df[pow_cols].mean(axis=1)
        pow_std_array = self.pw_curve_df[pow_cols].std(axis=1)

        ax.fill_between(
            x,
            np.array(pow_mean_array - 2 * pow_std_array),
            np.array(pow_mean_array + 2 * pow_std_array),
            color="tab:red",
            label="Uncertainty bounds (2 std. dev.)",
            alpha=0.30,
        )
        ax.plot(x, pow_mean_array, color="tab:red", label="Mean curve")

        if fm is not None:
            fm_turb = fm.core.farm.turbine_definitions[ti]
            ws_array = np.array(fm_turb["power_thrust_table"]["wind_speed"])
            pow_array = np.array(fm_turb["power_thrust_table"]["power"])
            ax.plot(ws_array, pow_array, "--", label="FLORIS curve")

        ax.legend()
        ax.set_title("Mean of all turbine power curves with UQ")
        return fig, ax

    def plot_filters_custom_scatter(
        self, ti, x_col, y_col, xlabel="Wind speed (m/s)", ylabel="Power (kW)", ax=None
    ):
        """Plot the filtered data in a scatter plot.

        Plot the filtered data in a scatter plot, categorized
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
            xlabel (str, optional): Figure x-axis label. Defaults to
                'Wind speed (m/s)'.
            ylabel (str, optional): Figure y-axis label. Defaults to
                'Power (kW)'.
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
            ids = df_f == flag
            df_subset = self._df_initial.loc[ids]
            percentage = 100.0 * np.sum(ids) / N
            if (
                any(ids)
                and (not df_subset[x_col].isna().all())
                and (not df_subset[y_col].isna().all())
            ):
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
        self._set_legend_alpha_to_one(lgd)

        if self.turbine_names is not None:
            ax.set_title(f"WTG {self.turbine_names[ti]}, [{ti:03d}]: Filters")
        else:
            ax.set_title("WTG {:03d}: Filters".format(ti))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        return ax

    def plot_filters_custom_scatter_bokeh(
        self,
        ti,
        x_col,
        y_col,
        title="Wind-speed vs. power curve",
        xlabel="Wind speed (m/s)",
        ylabel="Power (kW)",
        p=None,
    ):
        """Plot the filtered data in a scatter plot.

        Plot the filtered data in a scatter plot, categorized
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
            title (str, optional): Figure title. Defaults to 'Wind-
                speed vs. power curve'.
            xlabel (str, optional): Figure x-axis label. Defaults to
                'Wind speed (m/s)'.
            ylabel (str, optional): Figure y-axis label. Defaults to
                'Power (kW)'.
            p (Bokeh Figure, optional): Figure to plot in. If None is
                specified, creates a new figure. Defaults to None.

        Returns:
            ax: The figure axis in which the scatter plot is drawn.
        """
        # Create figure, if not specified

        bokeh_tooltips = [
            ("(x,y)", "($x, $y)"),
            ("time", "@time"),
            ("index", "$index"),
        ]

        if p is None:
            p = figure(
                title=title,
                width=800,
                height=550,
                sizing_mode="stretch_width",
                x_axis_label=xlabel,
                y_axis_label=ylabel,
                tooltips=bokeh_tooltips,
            )
            p.add_layout(Legend(title="Data category"), "right")

        # Get filter dataframe
        df_f = self.df_filters["WTG_{:03d}".format(ti)]
        all_flags = self._get_all_unique_flags()
        N = df_f.shape[0]

        # For each flagging condition, plot the results
        colors = itertools.cycle(palette)
        for flag in all_flags:
            ids = df_f == flag
            df_subset = self._df_initial.loc[ids]
            percentage = 100.0 * np.sum(ids) / N
            label = "{:s} ({:.2f} %)".format(flag, percentage)
            alpha = 0.65
            size = 5
            color = next(colors)
            if (
                any(ids)
                and (not df_subset[x_col].isna().all())
                and (not df_subset[y_col].isna().all())
            ):
                source = ColumnDataSource(
                    data=dict(
                        x=df_subset[x_col],
                        y=df_subset[y_col],
                        time=list(df_subset["time"].astype(str)),
                    )
                )
                p.circle(
                    "x",
                    "y",
                    source=source,
                    fill_alpha=alpha,
                    color=color,
                    line_color=None,
                    size=size,
                    legend_label=label,
                )

        p.legend.title = "Data category"
        p.legend.click_policy = "hide"
        p.toolbar.active_inspect = None

        return p

    def plot_filters_in_ws_power_curve(self, ti, fm=None, ax=None):
        """Plot faulty data in the wind speed power curve.

        Plot the wind speed power curve and connect each faulty datapoint
        to the label it was classified as faulty with.

        Args:
            ti (int): Turbine number which should be plotted.
            fm (FlorisModel, optional): floris object. If not None, will
            use this to plot the turbine power curves as implemented in floris.
            Defaults to None.
            ax (plt.Axis): Pyplot Axis object.

        Returns:
            ax: The figure axis in which the scatter plot is drawn.
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

        if fm is not None:
            fm_turb = fm.core.farm.turbine_definitions[ti]
            ws_array = np.array(fm_turb["power_thrust_table"]["wind_speed"])
            pow_array = np.array(fm_turb["power_thrust_table"]["power"])

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
        self._set_legend_alpha_to_one(lgd)

        if self.turbine_names is not None:
            ax.set_title(f"WTG {self.turbine_names[ti]}, [{ti:03d}]: Filters")
        else:
            ax.set_title("WTG {:03d}: Filters".format(ti))
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True)

        return ax

    def plot_postprocessed_in_ws_power_curve(self, ti, fm=None, ax=None):
        """Plot the postprocessed data in the wind speed power curve.

        Plot the wind speed power curve and mark faulty data according to
        their filters.

        Args:
            ti (int): Turbine number which should be plotted.
            fm (FlorisModel, optional): floris object. If not None, will
            use this to plot the turbine power curves as implemented in floris.
            Defaults to None.
            ax (Matplotlib.pyplot Axis, optional): Axis to plot in. If None is
               specified, creates a new figure and axis. Defaults to None.

        Returns:
            ax: The figure axis in which the scatter plot is drawn.
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

        if fm is not None:
            fm_turb = fm.core.farm.turbine_definitions[ti]
            ws_array = np.array(fm_turb["power_thrust_table"]["wind_speed"])
            pow_array = np.array(fm_turb["power_thrust_table"]["power"])

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
        self._set_legend_alpha_to_one(lgd)

        if self.turbine_names is not None:
            ax.set_title(f"WTG {self.turbine_names[ti]}, [{ti:03d}]: Postprocessed dataset")
        else:
            ax.set_title("WTG {:03d}: Postprocessed dataset".format(ti))
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True)

        return ax

    def plot_filters_in_time(self, ti, ax=None):
        """Plot the filtered data in time.

        Generate bar plot where each week of data is gathered and its
        filtering results will be shown relative to the data size of each
        week. This plot can particularly be useful to investigate whether
        certain weeks/time periods show a particular high number of faulty
        measurements. This can often be correlated with maintenance time
        windows and the user may opt to completely remove any measurements
        in the found time period from the dataset.

        Args:
            ti (int): Index of the turbine of interest.
            ax (Matplotlib.pyplot Axis, optional): Axis to plot in. If None is
               specified, creates a new figure and axis. Defaults to None.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(13, 7))

        # Get a list of all flags and then get colors correspondingly
        all_flags = self._get_all_unique_flags()

        # Manipulate dataframe to easily plot results
        df_f = self.df_filters["WTG_{:03d}".format(ti)]
        df_conditional = pd.concat(
            [pd.DataFrame({flag: np.array(df_f == flag, dtype=int)}) for flag in all_flags], axis=1
        )
        df_merged = pd.concat([df_conditional, self.df["time"]], axis=1)
        df_histogram = df_merged.groupby(
            [df_merged["time"].dt.year, df_merged["time"].dt.isocalendar().week]
        ).sum(numeric_only=True)

        # Plot the histogram information
        ax = df_histogram.plot.bar(stacked=True, ax=ax)
        ax.set_ylabel("Count (-)")
        if self.turbine_names is not None:
            ax.set_title(f"WTG {self.turbine_names[ti]}, [{ti:03d}]")
        else:
            ax.set_title("WTG {:03d}".format(ti))

        ax.grid(True)

        return ax

    def plot_filters_in_time_bokeh(self, ti, p=None):
        """Plot the filtered data in time.

        Generate bar plot where each week of data is gathered and its
        filtering results will be shown relative to the data size of each
        week. This plot can particularly be useful to investigate whether
        certain weeks/time periods show a particular high number of faulty
        measurements. This can often be correlated with maintenance time
        windows and the user may opt to completely remove any measurements
        in the found time period from the dataset.

        Args:
            ti (int): Index of the turbine of interest.
            p (Bokeh Figure, optional): Figure to plot in. If None is
               specified, creates a new figure. Defaults to None.

        Returns:
            axis: The figure axis in which the scatter plot is
        """
        if p is None:
            p = figure(
                title="Filters over time",
                width=800,
                height=550,
                sizing_mode="stretch_width",
                x_axis_label="Time (year - week)",
                y_axis_label="Number of data points (-)",
                # tooltips=bokeh_tooltips,
            )
            p.add_layout(Legend(title="Data category"), "right")

        # Get a list of all flags and then get colors correspondingly
        all_flags = self._get_all_unique_flags()

        # Manipulate dataframe to easily plot results
        df_f = self.df_filters["WTG_{:03d}".format(ti)]
        df_conditional = pd.concat(
            [pd.DataFrame({flag: np.array(df_f == flag, dtype=int)}) for flag in all_flags], axis=1
        )
        df_merged = pd.concat([df_conditional, self.df["time"]], axis=1)
        df_histogram = df_merged.groupby(
            [df_merged["time"].dt.year, df_merged["time"].dt.isocalendar().week]
        ).sum(numeric_only=True)

        filter_flags = list(df_histogram.columns)
        xlabels = [f"{year}-{week}" for year, week in df_histogram.index]
        x = np.arange(len(list(df_histogram.index)))

        heights = np.zeros(len(x), dtype=int)
        colors = itertools.cycle(palette)
        for f in filter_flags:
            y = np.array(df_histogram[f], dtype=int)
            p.vbar(
                x=x, bottom=heights, top=heights + y, width=0.7, legend_label=f, color=next(colors)
            )
            heights = heights + y

        # Format x-axis
        p.xaxis.major_label_orientation = np.pi / 2.0
        p.xaxis.ticker = x
        p.xaxis.major_label_overrides = dict(zip(x, xlabels))

        # Format legend and allow hide/show functionality
        p.legend.title = "Filter"
        p.legend.click_policy = "hide"

        return p


def filter_df_by_faulty_impacting_turbines(df, ti, df_impacting_turbines, verbose=True):
    """Assign faulty measurements based on upstream turbines faults.

    Assigns a turbine's measurement to NaN for each timestamp for which any of the turbines
      that are shedding a wake on this turbine is reporting NaN measurements.

    Args:
        df (pd.DataFrame | FlascDataFrame): Dataframe with SCADA data with measurements
            formatted according to wd_000, wd_001, wd_002, pow_000, pow_001,
            pow_002, and so on.
        ti (int): Turbine number for which we are filtering the data.
            Basically, each turbine that impacts that power production of
            turbine 'ti' by more than 0.1% is required to be reporting a
            non-faulty measurement. If not, we classify the measurement of
            turbine 'ti' as faulty because we cannot sufficiently know the
            inflow conditions of this turbine.
        df_impacting_turbines (pd.DataFrame): A Pandas DataFrame in the
        format of:

                               0       1          2   3   4   5   6
                wd
                0.0       [6, 5]     [5]     [3, 5]  []  []  []  []
                3.0          [6]     [5]     [3, 5]  []  []  []  []
                ...          ...     ...        ...  ..  ..  ..  ..
                354.0  [6, 5, 3]  [5, 0]     [3, 5]  []  []  []  []
                357.0     [6, 5]     [5]  [3, 5, 4]  []  []  []  []

        The columns indicate the turbine of interest, i.e., the turbine that
        is waked, and each row shows which turbines are waking that turbine
        for that particular wind direction ('wd'). Typically calculated using:

            import flasc.utilities.floris_tools as ftools
            df_impacting_turbines = ftools.get_all_impacting_turbines(fi)

        verbose (bool, optional): Print information to the console. Defaults
        to True.

    Returns:
        pd.DataFrame: The postprocessed dataframe for 'df', filtered for
        inter-turbine issues like curtailment and turbine downtime.
    """
    # Get number of turbines
    n_turbines = dfm.get_num_turbines(df)

    # Drop all measurements where an upstream turbine is affecting this turbine but also has
    # a NaN measurement itself
    ws_cols = ["ws_{:03d}".format(ti) for ti in range(n_turbines)]
    pow_cols = ["pow_{:03d}".format(ti) for ti in range(n_turbines)]

    # Get array of which turbines affect our test turbine
    wd_array = df["wd"]

    # Create interpolant returning impacting turbines
    xp = np.array(df_impacting_turbines[ti].index, dtype=float)
    fp = np.arange(len(xp), dtype=int)

    # Copy values over from 0 to 360 deg
    if (np.abs(xp[0]) < 0.001) & (np.max(xp) < 360.0):
        xp = np.hstack([xp, 360.0])
        fp = np.hstack([fp, fp[0]])

    # Get nearest neighbor indices
    f = interp1d(x=xp, y=fp, kind="nearest")

    ids = np.array(f(wd_array), dtype=int)
    turbines_impacted = df_impacting_turbines[ti].values[ids]

    # Organize as matrix for easy manipulations
    impacting_turbines_matrix = np.zeros((len(wd_array), n_turbines), dtype=bool)
    for ii, turbines_impacted_onewd in enumerate(turbines_impacted):
        impacting_turbines_matrix[ii, turbines_impacted_onewd] = True

    # Calculate True/False statement whether any of the turbines shedding a wake on our test_turbine
    # has a NaN ws or pow measurement
    test_turbine_impacted_by_nan_ws = np.any(
        np.isnan(np.array(df[ws_cols], dtype=float)) & impacting_turbines_matrix, axis=1
    )
    test_turbine_impacted_by_nan_pow = np.any(
        np.isnan(np.array(df[pow_cols], dtype=float)) & impacting_turbines_matrix, axis=1
    )
    test_turbine_impacted = test_turbine_impacted_by_nan_ws | test_turbine_impacted_by_nan_pow

    # Assign test turbine's measurements to NaN if any turbine that is waking this turbine
    # is reporting NaN measurements
    N_pre = df_get_no_faulty_measurements(df, ti)
    df_out = df_mark_turbdata_as_faulty(
        df=df,
        cond=test_turbine_impacted,
        turbine_list=[ti],
    )
    N_post = df_get_no_faulty_measurements(df_out, ti)

    if verbose:
        logger.info(
            "Faulty measurements for WTG {:02d} increased from {:.3f} % to {:.3f} %. Reason: "
            "'Turbine is impacted by faulty upstream turbine'.".format(
                ti, 100.0 * N_pre / df.shape[0], 100.0 * N_post / df.shape[0]
            )
        )

    return df_out
