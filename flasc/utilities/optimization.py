"""Module for optimization functions."""

import copy
from datetime import timedelta as td

import numpy as np
import scipy.optimize as opt
import scipy.stats as spst
from floris.utilities import wrap_180, wrap_360
from pandas.errors import DataError

from flasc.data_processing import time_operations as fsato
from flasc.logging_manager import LoggingManager
from flasc.utilities import circular_statistics as css, floris_tools as ftools, utilities as fsut

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


def find_timeshift_between_dfs(
    df1,
    df2,
    cols_df1,
    cols_df2,
    use_circular_statistics=False,
    t_step=np.timedelta64(30 * 24, "h"),
    correct_y_shift=False,
    y_shift_range=np.arange(-180.0, 180.0, 2.0),
    opt_bounds=None,
    opt_Ns=None,
    verbose=True,
):
    """Find the optimal time shift between two dataframes.

    This function is used to find the optimal time shift between two dataframes
    'df1' and 'df2'. The function will resample the dataframes to a common time
    vector and then compare the dataframes in time steps of 't_step'. The function
    will then find the optimal time shift between the two dataframes by minimizing
    the Pearson correlation coefficient between the two dataframes. The function
    can also correct for a bias in the y-values of the dataframes by minimizing
    the Pearson correlation coefficient between the two dataframes after a bias
    has been removed.

    Args:
        df1 (pd.DataFrame | FlascDataFrame): Dataframe 1.
        df2 (pd.DataFrame | FlascDataFrame): Dataframe 2.
        cols_df1 (list): Columns to use in dataframe 1.
        cols_df2 (list): Columns to use in dataframe 2.
        use_circular_statistics (bool): Use circular statistics for averaging.
            Default is False.
        t_step (np.timedelta64): Time step for comparison.
            Default is 30 days.
        correct_y_shift (bool): Correct for a bias in the y-values.
            Default is False.
        y_shift_range (np.array): Range of y-shifts to evaluate.
            Default is np.arange(-180.0, 180.0, 2.0).
        opt_bounds (list): Bounds for optimization.
            Default is None.
        opt_Ns (int): Number of steps for optimization.
            Default is None.
        verbose (bool): Print verbose output.
            Default is True.

    Returns:
        list: List of dictionaries with the following keys:
            - t0: Start time of comparison.
            - t1: End time of comparison.
            - x_opt: Optimal time shift.
            - J_opt: Optimal cost function value.
            - x: Time shifts evaluated.
            - J: Cost function values evaluated.

    """
    if np.any(df1["time"].diff() < td(seconds=0)):
        raise DataError("Dataframe 1 not sorted by time.")

    if np.any(df2["time"].diff() < td(seconds=0)):
        raise DataError("Dataframe 2 not sorted by time.")

    # Deal with incompatible inputs
    if (correct_y_shift) and (not use_circular_statistics):
        raise NotImplementedError("Incompatible input specified.")

    # Get min and max time for dataframes
    min_time = np.datetime64(np.max([df1.head(1)["time"], df2.head(1)["time"]]))
    max_time = np.datetime64(np.min([df1.tail(1)["time"], df2.tail(1)["time"]]))

    # Convert to arrays of a single mean quantity
    logger.info("Determining one column-average per dataframe to sync.")
    if use_circular_statistics:
        df1["y1"] = css.calc_wd_mean_radial(df1[cols_df1], axis=1)
        df2["y2"] = css.calc_wd_mean_radial(df2[cols_df2], axis=1)
    else:
        df1["y1"] = np.nanmean(df1[cols_df1], axis=1)
        df2["y2"] = np.nanmean(df2[cols_df2], axis=1)

    # Cut down df1 and df2 to a minimal dataframe
    df1 = df1[["time", "y1"]]
    df2 = df2[["time", "y2"]]

    # Create a shared time array and sync both dfs on it
    try:
        # Try estimation of dt from first 5k entries
        dt = np.min(
            [fsut.estimate_dt(df1.iloc[0:5000]["time"]), fsut.estimate_dt(df2.iloc[0:5000]["time"])]
        )
    except (ValueError, TypeError) as e:
        # Handle specific exception
        logger.info(f"Error estimating dt: {e}")
        # Use full dataframe if fails somehow
        dt = np.min([fsut.estimate_dt(df1["time"]), fsut.estimate_dt(df2["time"])])

    logger.info("Resampling dataframes to a common time vector. " + "This may take a while...")
    time_array = min_time + np.arange(0, max_time - min_time + np.timedelta64(dt), dt)
    max_gap = 1.5 * dt
    df1 = fsato.df_resample_by_interpolation(
        df1,
        time_array=time_array,
        circular_cols=use_circular_statistics,
        interp_method="linear",
        max_gap=max_gap,
    )
    df2 = fsato.df_resample_by_interpolation(
        df2,
        time_array=time_array,
        circular_cols=use_circular_statistics,
        interp_method="linear",
        max_gap=max_gap,
    )

    # Look at comparison per t_step
    current_time = min_time
    output_list = []
    logger.info("Estimating required timeshift for df1.")
    while current_time < max_time:
        t0 = np.array(current_time, dtype="datetime64")
        t1 = np.array(np.datetime64(current_time) + np.timedelta64(t_step), dtype="datetime64")
        id_sub = (df1.time >= t0) & (df1.time < t1)
        df1_sub = df1[id_sub]
        df2_sub = df2[id_sub]

        # Create x, y1 and y2 vectors
        x = np.array(df1_sub["time"], dtype=np.datetime64)
        x = np.array((x - x[0]) / np.timedelta64(1, "s"))
        y1 = np.array(df1_sub["y1"])
        y2 = np.array(df2_sub["y2"])

        if verbose:
            logger.info("   Calculating timeshift for t0: %s, t1: %s" % (str(t0), str(t1)))

        def cost_fun(x_shift):
            # Shift data along x-axis and then fit along y-axis, if necessary
            y1_cor = fsut.interp_with_max_gap(x, x - x_shift, y1, max_gap=max_gap, kind="linear")

            if correct_y_shift:
                y_bias, J = match_y_curves_by_offset(
                    y1_cor, y2, dy_eval=y_shift_range, angle_wrapping=use_circular_statistics
                )
                y1_cor = y1_cor - y_bias
                if use_circular_statistics:
                    y1_cor = wrap_360(y1_cor)

            # Remove NaNs and infs
            ids = (~np.isnan(y1_cor)) & (~np.isnan(y2)) & (~np.isinf(y1_cor)) & (~np.isinf(y2))
            y2_cor = y2[ids]
            y1_cor = y1_cor[ids]

            # Calculate score
            if (len(y1_cor) < 10) | (len(y2_cor) < 10):
                cost = np.nan
            else:
                cost = -1.0 * spst.pearsonr(y1_cor, y2_cor)[0]
            return cost

        # Optimize using scipy.brute()
        if opt_bounds is None:
            opt_bounds = [(-td(hours=10), td(hours=10))]
        elif isinstance(opt_bounds[0], (tuple, list)):
            opt_bounds = opt_bounds  # Fine
        else:
            opt_bounds = [opt_bounds]

        if isinstance(opt_bounds[0][0], np.timedelta64):
            opt_bounds[0] = (
                opt_bounds[0][0] / np.timedelta64(1, "s"),
                opt_bounds[0][1] / np.timedelta64(1, "s"),
            )
        if isinstance(opt_bounds[0][0], td):
            opt_bounds[0] = (opt_bounds[0][0] / td(seconds=1), opt_bounds[0][1] / td(seconds=1))

        if opt_Ns is None:
            # Explore in steps of 10 minutes
            opt_Ns = int((opt_bounds[0][1] - opt_bounds[0][0]) / 600.0)

        finish = opt.fmin  # Can also be None
        if verbose:
            opt_disp = "iter"
        else:
            opt_disp = "final"
        x_opt, J_opt, x_all, J_all = opt.brute(
            cost_fun, ranges=opt_bounds, Ns=opt_Ns, finish=finish, disp=opt_disp, full_output=True
        )
        logger.info(
            "     Optimal time shift for df_1: %d s (%.2f hours)." % (x_opt[0], x_opt[0] / 3600.0)
        )

        output_list.append(
            {
                "t0": t0,
                "t1": t1,
                "x_opt": td(seconds=x_opt[0]),
                "J_opt": J_opt,
                "x": x_all,
                "J": J_all,
            }
        )

        current_time = np.datetime64(current_time) + np.timedelta64(t_step)

    return output_list


# def find_bias_x(x_1, y_1, x_2, y_2, search_range, search_dx):
#     x_1 = np.array(x_1)
#     y_2 = np.array(y_2)

#     def errfunc(dx, x_1, x_2, y_1, y_2):
#         y_1_cor = np.interp(x_2, x_1 - dx, y_1)

#         # Clean up data
#         clean_data = (~np.isnan(y_1_cor)) & (~np.isnan(y_2))
#         y_1_cor = y_1_cor[clean_data]
#         y_2 = y_2[clean_data]

#         if all(np.isnan(y_1_cor)) and all(np.isnan(y_2)):
#             cost = np.nan
#         else:
#             cost = -1.0 * spst.pearsonr(y_1_cor, y_2)[0]
#         return cost

#     cost_min = 1.0e15
#     success = False
#     p1 = [0.0]
#     for dx_opt in np.arange(search_range[0], search_range[1], search_dx):
#         cost_eval = errfunc(dx_opt, x_1, x_2, y_1, y_2)
#         if cost_eval <= cost_min:
#             p1 = [dx_opt]
#             cost_min = cost_eval
#             success = True
#     dx_opt = p1[0]

#     return dx_opt, success


def match_y_curves_by_offset(yref, ytest, dy_eval=None, angle_wrapping=True):
    """Match two curves by finding the optimal offset.

    This function is used to match two curves by finding the optimal offset
    between the two curves. The function will minimize the mean squared error
    between the two curves for a range of offsets.

    Args:
        yref (np.array): Reference curve.
        ytest (np.array): Test curve.
        dy_eval (np.array): Range of offsets to evaluate.
            Default is None.
        angle_wrapping (bool): Use angle wrapping for evaluation.
            Default is True.

    Returns:
        Tuple (float, float) with the following elements:
            dwd_opt: Optimal offset.
            J_opt: Optimal cost function value.

    """
    if dy_eval is None:
        if angle_wrapping:
            dy_eval = np.arange(-180.0, 180.0, 2.0)
        else:
            raise ValueError("Requires dy_eval if wrap_360==False.")

    J_opt = np.nan
    dwd_opt = np.nan
    for dy in dy_eval:
        if angle_wrapping:
            ytest_cor = wrap_360(ytest - dy)
            y_error = np.abs(wrap_180(yref - ytest_cor))
        else:
            ytest_cor = ytest - dy
            y_error = np.abs(yref - ytest_cor)

        if np.all(np.isnan(y_error)) | (len(y_error) < 1):
            J = np.nan
        else:
            J = np.nanmean(y_error**2.0)

        if np.isnan(J_opt):
            if not np.isnan(J):
                J_opt = J
                dwd_opt = dy
        elif J < J_opt:
            J_opt = J
            dwd_opt = dy
    return dwd_opt, J_opt


def estimate_ti(
    fm,
    P_measured,
    Ns,
    bounds,
    turbine_upstream,
    turbines_downstream,
    refine_with_fmin=False,
    verbose=False,
):
    """Estimate the turbulence intensity for a given turbine.

    This function is used to estimate the turbulence intensity for a given
    turbine.

    Args:
        fm (FlorisModel): FlorisModel object.
        P_measured (np.array): Measured power data.
        Ns (int): Number of steps for optimization.
        bounds (list): Bounds for optimization.
        turbine_upstream (int): Upstream turbine.
        turbines_downstream (list): Downstream turbines.
        refine_with_fmin (bool): Refine with fmin.
            Default is False.
        verbose (bool): Print verbose output.
            Default is False.

    Returns:
        dict: Dictionary with the following
            keys:
            - x_opt: Optimal time shift.
            - J_opt: Optimal cost function value.
            - x: Time shifts evaluated.
            - J: Cost function values evaluated.

    """
    # Make copy so that existing object is not changed
    fm = copy.deepcopy(fm)
    num_turbines = len(fm.layout_x)
    ti_0 = np.mean(fm.core.farm.turbulence_intensity)

    # Define a cost function
    def cost_fun(ti):
        ti_array = np.repeat(ti_0, num_turbines)
        ti_array[turbine_upstream] = ti
        ftools._fi_set_ws_wd_ti(fm, ti=ti_array)
        fm.run()
        Pturbs = np.array(fm.get_turbine_power())
        Pturbs = Pturbs[turbines_downstream]
        se = (P_measured - Pturbs) ** 2.0
        mse = np.mean(se)
        return mse

    if refine_with_fmin:
        finish = opt.fmin
    else:
        finish = None

    # Ensure appropriate format
    if not (isinstance(bounds[0], tuple) | isinstance(bounds[0], list)):
        bounds = [bounds]

    # Optimize using grid search approach
    x_opt, J_opt, x, J = opt.brute(
        cost_fun, ranges=bounds, Ns=Ns, finish=finish, disp=verbose, full_output=True
    )

    opt_result = {"x_opt": x_opt, "J_opt": J_opt, "x": x, "J": J}
    return opt_result
