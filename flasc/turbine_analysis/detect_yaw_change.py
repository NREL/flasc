# Copyright 2021 NREL and SHELL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floris.utilities import wrap_180, wrap_360

from flasc.circular_statistics import calc_wd_mean_radial
from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc.turbine_analysis import ws_pow_filtering as wspf


def filter_min_power(df, min_power):
    ws_pow_filtering = wspf.ws_pw_curve_filtering(df=df)
    n_turbines = dfm.get_num_turbines(df)

    for ti in range(n_turbines):
        # Filter for NaN wind speed or power productions
        ws_pow_filtering.filter_by_condition(
            condition=(ws_pow_filtering.df["pow_{:03d}".format(ti)] < min_power),
            label="Lower Power",
            ti=ti,
            verbose=True,
        )
    df = ws_pow_filtering.get_df()

    return df


def remove_yaw_shifts(
    df_in,
    shift_threshold_deg=3.0,
    min_power=None,
    turbines_to_check=None,
    turbines_to_reference=None,
    plot_change=False,
):
    # Get a local copy and an original version
    df = df_in.copy(deep=True)
    df_original = df_in.copy(deep=True)

    # Get the number of turbines
    n_turbines = dfm.get_num_turbines(df)

    # Check if yaw_deshift_000 in columns, if so then assume this is a 2nd, 3rd pass
    if "yaw_deshift_000" in df_in.columns:
        yaw_stub = "yaw_deshift"
    else:
        yaw_stub = "yaw"

    # If turbines to check is empty, assume it is all
    if turbines_to_check is None:
        turbines_to_check = list(range(n_turbines))
    turbines_to_check_names = [f"{yaw_stub}_{t_idx:03d}" for t_idx in turbines_to_check]

    # If turbines to check is empty, assume it is all
    if turbines_to_reference is None:
        turbines_to_reference = list(range(n_turbines))
    turbines_to_reference_names = [f"{yaw_stub}_{t_idx:03d}" for t_idx in turbines_to_reference]

    # All turbines is the combination of both
    all_turbines = list(set(turbines_to_check).union(set(turbines_to_reference)))
    all_turbines_names = [f"{yaw_stub}_{t_idx:03d}" for t_idx in all_turbines]

    # If power_min is not None, apply a filter on minimum power
    if min_power is not None:
        df = filter_min_power(df, min_power)

    # Limit down to yaw angles of all turbines to reference and check
    df = df[all_turbines_names]

    # For each turbine in turbines_to_check, compute an error term,
    # and a cumsum of that error, and identify the shift
    for t_name in turbines_to_check_names:
        # Add the circular mean of all the columns except the present turbine
        circ_mean_turbines = [n for n in turbines_to_reference_names if n != t_name]
        df["circmean"] = calc_wd_mean_radial(df[circ_mean_turbines], axis=1)

        # Get a deshifted name
        deshift_name = (t_name + "_deshift").replace("deshift_deshift", "deshift")

        df[t_name + "_error"] = wrap_180(df[t_name] - df["circmean"])

        # Remove the mean from the error
        df[t_name + "_error"] = df[t_name + "_error"] - df[t_name + "_error"].mean()

        df[t_name + "_error_cumsum"] = df[t_name + "_error"].cumsum().abs()

        # The proposed shift point is the max index of the abs cumsum
        max_cumsum_index = df[t_name + "_error_cumsum"].idxmax()
        max_cumsum_index_iloc = df.index.get_loc(max_cumsum_index)

        # if the max is essentially at the end
        if (df.shape[0] - max_cumsum_index_iloc < 10) or (max_cumsum_index_iloc < 2):
            change_deg = 0.0

        else:
            # Calculating the average of non-NaN values before and after the specified index
            average_before = df.iloc[: max_cumsum_index_iloc - 1][t_name + "_error"].mean()
            average_after = df.iloc[max_cumsum_index_iloc + 1 :][t_name + "_error"].mean()
            change_deg = average_after - average_before

        if np.abs(change_deg) < shift_threshold_deg:
            print(
                f"The change after the shift in {t_name} is {change_deg:.02} deg"
                f" which is less than the threshold of {shift_threshold_deg} deg"
                f", assigning no change to {deshift_name}"
            )
            df_original[deshift_name] = df[t_name]

        else:
            print(f"{t_name} yaw shifts by {change_deg:.02} deg")

            # Apply the deshifted value following the change point
            values_to_shift = df[t_name].values.copy()
            values_to_shift[max_cumsum_index_iloc:] = wrap_360(
                values_to_shift[max_cumsum_index_iloc:] - change_deg
            )
            df_original[deshift_name] = values_to_shift
            df[deshift_name] = values_to_shift

            # Recalculate the error
            df[t_name + "_error_deshift"] = wrap_180(df[deshift_name] - df["circmean"])
            df[t_name + "_error_deshift"] = (
                df[t_name + "_error_deshift"] - df[t_name + "_error_deshift"].mean()
            )

            # If reqested, plot the change
            if plot_change:
                fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

                # Show the change in error
                ax = axarr[0]
                ax.plot(df_original["time"], df[t_name + "_error"], "r--", label="Original Error")
                ax.plot(
                    df_original["time"],
                    df[t_name + "_error_deshift"],
                    "b--",
                    label="Original Error",
                )
                ax.legend()
                ax.grid(True)

                # Compare the yaw signal to the average before and after the shift
                ax = axarr[1]
                ax.plot(df_original["time"], df_original[t_name], "r--", label="Original Yaw")

                ax.plot(
                    df_original["time"], df_original[deshift_name], "b", label="New value", lw=2
                )
                ax.plot(df_original["time"], df["circmean"], "k--", label="Average of refs")
                ax.legend()
                ax.grid(True)

                # Finally show the cumsum signal
                ax = axarr[2]
                ax.plot(df_original["time"], df[t_name + "_error_cumsum"], "k", label="CUMSUM")
                ax.axvline(df_original["time"].values[max_cumsum_index_iloc], color="r")
                ax.legend()
                ax.grid(True)

    return df_original, df


if __name__ == "__main__":
    # Generate a set of quick fake data for testing
    num_turbines = 3
    num_minutes = 60

    # Generate the signals
    time = np.arange(num_minutes)
    p_series = np.ones(num_minutes) * 200
    y_series = np.ones(num_minutes) * 270

    for i in range(1, num_minutes):
        p_series[i] = p_series[i - 1] + np.random.random() * 5
        y_series[i] = p_series[i - 1] + np.random.random() * 5

    # Wrap
    y_series = wrap_360(y_series)

    df = pd.DataFrame({"time": time})

    for t_idx in range(num_turbines):
        df["pow_%03d" % t_idx] = p_series + np.random.random(len(p_series)) * 5
    for t_idx in range(num_turbines):
        df["yaw_%03d" % t_idx] = wrap_360(y_series + np.random.random(len(p_series)) * 2)

    # Introduce a  5 deg shift in the 0th turbine yaw at the halfway point
    yaw_000_values = df["yaw_000"].values.copy()
    yaw_000_values[int(num_minutes / 2) :] = yaw_000_values[int(num_minutes / 2) :] + 5
    df["yaw_000"] = yaw_000_values

    df, _ = remove_yaw_shifts(df, plot_change=True)

    print(df.head())

    plt.show()
