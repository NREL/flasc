# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
from datetime import timedelta as td

import numpy as np
import pandas as pd

from floris.tools.floris_interface import FlorisInterface
from floris.utilities import wrap_360

from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc import floris_tools as ftls


def get_wind_data_from_nwtc():
    """Function that downloads WD, WS and TI data from a measurement
    tower from NRELs wind technology center (NWTC) site. This data
    is sampled at 60 s.

    Returns:
        df [pd.DataFrame]: Dataframe with time, WD, WS and TI for one year
        of measurement tower data, sampled at 60 s.
    """
    import urllib.request

    root_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root_dir, "nwtc_2019_2020_rawdata.csv")

    if not os.path.exists(csv_path):
        print("File not found. Beginning file download with urllib2...")
        url = (
            "https://midcdmz.nrel.gov/apps/plot.pl?site=NWTC&start=20010824&"
            + "edy=28&emo=9&eyr=2021&year=2019&month=1&day=1&endyear=2020&"
            + "endmonth=1&endday=1&time=0&inst=25&inst=37&inst=43&inst=55&"
            + "inst=62&type=data&wrlevel=2&preset=0&first=3&math=0&second=-1"
            + "&value=0.0&user=0&axis=1"
        )
        urllib.request.urlretrieve(url, os.path.join(root_dir, csv_path))

    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    # Determine a wind rose based on met mast data from NWTC
    print("=================================================================")
    print("Downloading and importing met mast data from the NWTC website")
    df = get_wind_data_from_nwtc()

    # Filter to ensure ascending time and no time duplicates
    print("Sorting by time and localizing time entries to MST timezone")
    time_array = df["DATE (MM/DD/YYYY)"] + " " + df["MST"] + ":00"
    df["time"] = pd.to_datetime(time_array, format="%m/%d/%Y %H:%M:%S")
    df = (
        df.set_index("time", drop=True)
        .tz_localize(tz="GMT")
        .reset_index(drop=False)
    )

    # Sort dataframe by time and fix duplicates
    df = dfm.df_sort_and_fix_duplicates(df)

    # Initialize the FLORIS interface fi
    print("Initializing the FLORIS object for our demo wind farm")
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "demo_floris_input.yaml")
    fi = FlorisInterface(fi_path)

    # Format columns to generic names
    print("Formatting the dataframe with met mast data...")
    df = df.drop(columns=["DATE (MM/DD/YYYY)", "MST"])
    df = df.rename(
        columns={
            "Avg Wind Speed @ 80m [m/s]": "ws",
            "Avg Wind Speed (std dev) @ 80m [m/s]": "ws_std",
            "Avg Wind Direction @ 80m [deg]": "wd",
            "Avg Wind Direction (std dev) @ 80m [deg]": "wd_std",
            "Turbulence Intensity @ 80m": "ti",
        }
    )

    print(" ")
    print("=================================================================")
    print("Calculating FLORIS solutions. This may take in the order of 1 hr.")
    # First calculate FLORIS solutions for a predefined grid of amospheric
    # conditions, being wind speeds, wind directions and turb. intensities
    root_path = os.path.dirname(os.path.abspath(__file__))
    fn_approx = os.path.join(root_path, "df_approx.ftr")
    if os.path.exists(fn_approx):
        df_approx = pd.read_feather(fn_approx)
    else:
        df_approx = ftls.calc_floris_approx_table(
            fi=fi,
            wd_array=np.arange(0.0, 360.0, 3.0),
            ws_array=np.arange(0.001, 27.0, 1.0),
            ti_array=np.arange(0.03, 0.30, 0.03),
            save_turbine_inflow_conditions_to_df=True
        )
        df_approx.to_feather(fn_approx)

    # Now interpolate solutions from precalculated table to dataframe
    df_fi = ftls.interpolate_floris_from_df_approx(
        df=df, df_approx=df_approx, method="linear", verbose=True
    )

    # Add noise and bias per turbine
    np.random.seed(123)  # Fixed seed for reproducability
    for ti in range(len(fi.layout_x)):
        df_fi["pow_%03d" % ti] *= 1.0 + 0.03 * np.random.randn(df_fi.shape[0])
        df_fi["wd_{:03d}".format(ti)] += 3.0 * np.random.randn(df_fi.shape[0])
        df_fi["ws_%03d" % ti] += 0.2 * np.random.randn(df_fi.shape[0])
        df_fi["ti_%03d" % ti] += 0.01 * np.random.randn(df_fi.shape[0])

        # Saturate power to rated power after adding noise
        df_fi["pow_%03d" % ti] = np.vstack(
            [
                df_fi["pow_%03d" % ti].values,
                np.array([5000.0] * df_fi.shape[0]),
            ]
        ).min(axis=0)

        # Add realistic noise to ws_array-pow_array curve
        ws_array = np.array(df_fi["ws_{:03d}".format(ti)], dtype=float)
        pow_array = np.array(df_fi["pow_{:03d}".format(ti)], dtype=float)
        ids = (ws_array >= 12.5) & (pow_array >= 4980.0)
        pow_array[ids] = np.random.randint(
            low=4995, high=5005
        ) + 20 * np.random.rand(np.sum(ids))
        ids = np.where((ws_array >= 12.5) & (pow_array <= 4950.0))[0]
        ids = np.random.choice(ids, int(np.ceil(len(ids) * 0.97)), False)
        pow_array[ids] = np.random.randint(
            low=4995, high=5005
        ) + 5 * np.random.rand(len(ids))

        # Add samples: region 2.5
        ids = np.random.choice(
            np.where((ws_array >= 11.5) & (ws_array <= 14.2))[0],
            replace=False,
            size=300,
        )
        pow_array[ids] = 5010 - (14.5 - ws_array[ids]) * np.random.randint(
            0.0, 275.0, size=300
        )

        # Add samples: derating
        ids = np.random.choice(
            np.where((ws_array >= 10))[0], size=600, replace=False
        )
        ws_array[ids[0:300]] = np.random.randint(10166, 12000, size=300) / 1000
        ws_array[ids[300::]] = 11.166 + 7 * np.random.rand(300)
        pow_array[ids] = np.random.randint(
            3497, 3503, size=600
        ) + 5 * np.random.rand(600)

        # Add samples: above rated power
        ids = np.random.choice(
            np.where((ws_array >= 11))[0], size=300, replace=False
        )
        pow_array[ids] = np.random.randint(4995, 5080, size=300)

        # Add samples: noise
        ids = np.random.choice(
            np.where((ws_array >= 12.5) & (ws_array <= 25.0))[0],
            size=300,
            replace=False,
        )
        pow_array[ids] = np.random.randint(0, 4950, size=300)

        # Add samples: noise under curve in region 2
        ids = np.random.choice(
            np.where((ws_array >= 1.5) & (ws_array <= 12.2))[0],
            size=300,
            replace=False,
        )
        pow_array[ids] = (ws_array[ids] - 1.5) * np.random.randint(
            0.0, 5000.0 / 9.0, size=300
        )

        # Add samples: negative wind speeds and/or powers
        ids = np.random.choice(
            range(len(ws_array)),
            size=300,
            replace=False
        )
        ws_array[ids] = np.random.randint(-500, -10) / 1000
        pow_array[ids] = np.random.randint(-25000, -10) / 1000

        # Add samples: mark couple days/weeks of year as bad/maintenance
        t_start = df_fi.iloc[0]["time"] + td(days=np.random.randint(1, 360))
        t_end = t_start + td(days=np.random.randint(1, 9))

        ids = (df_fi["time"] >= t_start) & (df_fi["time"] < t_end)
        self_flag = np.ones(len(ws_array), dtype=bool)
        self_flag[ids] = False
        pow_array[ids] = pow_array[ids] * 0.01  # Random pertubation

        # Mark random data as bad and multiply them with a random number
        N = np.random.randint(0, int(np.ceil(0.05 * len(ws_array))))
        ids = np.random.choice(range(len(ws_array)), size=N)
        self_flag[ids] = False
        pow_array[ids] *= 0.01 * np.random.randint(0, 100, size=len(ids))

        # fig, ax = plt.subplots()
        # ax.plot(ws_array, pow_array, ".")
        # plt.show()

        # Mimic northing error
        df_fi["wd_{:03d}".format(ti)] += 50.0 * (np.random.rand() - 0.50)
        df_fi["wd_{:03d}".format(ti)] = wrap_360(df_fi["wd_{:03d}".format(ti)])

        # Mimic sensor stuck type of faults in ws
        N = np.random.randint(0, 15)
        ids = np.random.choice(range(len(ws_array)), size=N)
        for id in ids:
            for il in range(np.random.randint(5, 24)):
                ws_array[id+il] = ws_array[id]

        # Mimic sensor stuck type of faults in wd
        wd_array = df_fi["wd_{:03d}".format(ti)]
        N = np.random.randint(0, 15)
        ids = np.random.choice(range(len(wd_array)), size=N)
        for id in ids:
            for il in range(np.random.randint(5, 24)):
                wd_array[id+il] = wd_array[id]
        df_fi["wd_{:03d}".format(ti)] = wd_array

        # Mimic northing error
        wd_array = wrap_360(wd_array + 50.0 * (np.random.rand() - 0.50))

        # Save to self
        df_fi["wd_{:03d}".format(ti)] = wd_array
        df_fi["ws_{:03d}".format(ti)] = ws_array
        df_fi["pow_{:03d}".format(ti)] = pow_array
        df_fi["is_operation_normal_{:03d}".format(ti)] = self_flag

    # Add curtailment to two turbines for January until mid May
    df_fi.loc[
        (df_fi.index < 250000) & (df_fi["pow_002"] > 3150.0), "pow_002"
    ] = 3150.0
    df_fi.loc[
        (df_fi.index < 250000) & (df_fi["pow_005"] > 3150.0), "pow_005"
    ] = 3150.0

    # Rename true 'wd', 'ws_array' and 'ti' channels
    df_fi = df_fi.rename(
        columns={"ws": "ws_truth", "wd": "wd_truth", "ti": "ti_truth"}
    )

    # # Now rename variables and turbine names
    # tnames = ["A1", "A2", "A3", "B1", "B2", "C1", "C2"]
    # for ti in range(7):
    #     df_fi = df_fi.rename(columns={
    #         "ws_{:03d}".format(ti): "NacWSpeed_{:s}".format(tnames[ti]),
    #         "wd_{:03d}".format(ti): "NacWDir_{:s}".format(tnames[ti]),
    #         "ti_{:03d}".format(ti): "NacTI_{:s}".format(tnames[ti]),
    #         "pow_{:03d}".format(ti): "ActivePower_{:s}".format(tnames[ti]),
    #         "is_operation_normal_{:03d}".format(ti): \
    #             "is_operation_normal_{:s}".format(tnames[ti]),
    # })

    fout = os.path.join(root_path, "demo_dataset_scada_60s.ftr")
    df_fi.to_feather(fout)
    print("Saved artificial SCADA dataset to: '" + fout + "'.")

    # Now make a met mast dataset with gap in data, 2 hr time shift with SCADA
    # and additional noise.
    df_met_mast = df_fi[["time", "ws_truth", "wd_truth", "ti_truth"]].copy()
    ids = np.hstack([range(150000), range(210000, df_met_mast.shape[0])])
    df_met_mast = df_met_mast.iloc[ids].reset_index(drop=True)  # Mimic gap in data
    df_met_mast = df_met_mast.rename(columns={
        "wd_truth": "WindDirection_80m",
        "ws_truth": "WindSpeed_80m",
        "ti_truth": "TurbulenceIntensity_80m"
    })
    df_met_mast["WindDirection_80m"] = (
        wrap_360(df_met_mast["WindDirection_80m"] + np.random.randint(0, 6))
    )
    df_met_mast["time"] += td(hours=2)
    fout = os.path.join(root_path, "demo_dataset_metmast_60s.ftr")
    df_met_mast.to_feather(fout)
    print("Saved artificial met mast dataset to: '" + fout + "'.")
