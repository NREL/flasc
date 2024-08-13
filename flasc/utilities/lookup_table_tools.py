"""Module for creating interpolants from lookup tables."""

import numpy as np
from scipy.interpolate import LinearNDInterpolator


def get_yaw_angles_interpolant(
    df_opt, ramp_up_ws=[4, 5], ramp_down_ws=[10, 12], minimum_yaw_angle=None, maximum_yaw_angle=None
):
    """Get an interpolant for the optimal yaw angles from a dataframe.

    Create an interpolant for the optimal yaw angles from a dataframe
    'df_opt', which contains the rows 'wind_direction', 'wind_speed',
    'turbulence_intensity', and 'yaw_angles_opt'. This dataframe is typically
    produced automatically from a FLORIS yaw optimization using Serial Refine
    or SciPy. One can additionally apply a ramp-up and ramp-down region
    to transition between non-wake-steering and wake-steering operation.

    Args:
        df_opt (pd.DataFrame): Dataframe containing the rows 'wind_direction',
            'wind_speed', 'turbulence_intensity', and 'yaw_angles_opt'.
        ramp_up_ws (list, optional): List with length 2 depicting the wind
            speeds at which the ramp starts and ends, respectively, on the lower
            end. This variable defaults to [4, 5], meaning that the yaw offsets are
            zero at and below 4 m/s, then linearly transition to their full offsets
            at 5 m/s, and continue to be their full offsets past 5 m/s. Defaults to
            [4, 5].
        ramp_down_ws (list, optional): List with length 2 depicting the wind
            speeds at which the ramp starts and ends, respectively, on the higher
            end. This variable defaults to [10, 12], meaning that the yaw offsets are
            full at and below 10 m/s, then linearly transition to zero offsets
            at 12 m/s, and continue to be zero past 12 m/s. Defaults to [10, 12].
        minimum_yaw_angle (float, optional): The minimum yaw angle in degrees.
            Defaults to None.  If None, the minimum yaw angle is set to the minimum
            yaw angle in the dataset.
        maximum_yaw_angle (float, optional): The maximum yaw angle in degrees.
            Defaults to None.  If None, the maximum yaw angle is set to the maximum
            yaw angle in the dataset.

    Returns:
        LinearNDInterpolator: An interpolant function which takes the inputs
            (wind_directions, wind_speeds, turbulence_intensities), all of equal
            dimensions, and returns the yaw angles for all turbines. This function
            incorporates the ramp-up and ramp-down regions.
    """
    # Load data and set up a linear interpolant
    points = df_opt[["wind_direction", "wind_speed", "turbulence_intensity"]]
    values = np.vstack(df_opt["yaw_angles_opt"])

    # Derive maximum and minimum yaw angle (deg)
    if minimum_yaw_angle is None:
        minimum_yaw_angle = np.min(values)
    if maximum_yaw_angle is None:
        maximum_yaw_angle = np.max(values)

    # Expand wind direction range to cover 0 deg to 360 deg
    points_copied = points[points["wind_direction"] == 0.0].copy()
    points_copied.loc[points_copied.index, "wind_direction"] = 360.0
    values_copied = values[points["wind_direction"] == 0.0, :]
    points = np.vstack([points, points_copied])
    values = np.vstack([values, values_copied])

    # Copy lowest wind speed / TI solutions to -1.0 to create lower bound
    for col in [1, 2]:
        ids_to_copy_lb = points[:, col] == np.min(points[:, col])
        points_copied = np.array(points[ids_to_copy_lb, :], copy=True)
        values_copied = np.array(values[ids_to_copy_lb, :], copy=True)
        points_copied[:, col] = -1.0  # Lower bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

        # Copy highest wind speed / TI solutions to 999.0
        ids_to_copy_ub = points[:, col] == np.max(points[:, col])
        points_copied = np.array(points[ids_to_copy_ub, :], copy=True)
        values_copied = np.array(values[ids_to_copy_ub, :], copy=True)
        points_copied[:, col] = 999.0  # Upper bound
        points = np.vstack([points, points_copied])
        values = np.vstack([values, values_copied])

    # Now create a linear interpolant for the yaw angles
    interpolant = LinearNDInterpolator(points=points, values=values, fill_value=np.nan)

    # Now create a wrapper function with ramp-up and ramp-down
    def interpolant_with_ramps(wd_array, ws_array, ti_array=None):
        # Deal with missing ti_array
        if ti_array is None:
            ti_ref = float(np.median(interpolant.points[:, 2]))
            ti_array = np.ones(np.shape(wd_array), dtype=float) * ti_ref

        # Format inputs
        wd_array = np.array(wd_array, dtype=float)
        ws_array = np.array(ws_array, dtype=float)
        ti_array = np.array(ti_array, dtype=float)
        yaw_angles = interpolant(wd_array, ws_array, ti_array)
        yaw_angles = np.array(yaw_angles, dtype=float)

        # Define ramp down factor
        rampdown_factor = np.interp(
            x=ws_array,
            xp=[0.0, *ramp_up_ws, *ramp_down_ws, 999.0],
            fp=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        )

        # Saturate yaw offsets to threshold
        axis = len(np.shape(yaw_angles)) - 1
        nturbs = np.shape(yaw_angles)[-1]
        yaw_lb = np.expand_dims(minimum_yaw_angle * rampdown_factor, axis=axis).repeat(
            nturbs, axis=axis
        )
        yaw_ub = np.expand_dims(maximum_yaw_angle * rampdown_factor, axis=axis).repeat(
            nturbs, axis=axis
        )

        return np.clip(yaw_angles, yaw_lb, yaw_ub)

    return interpolant_with_ramps
