# Copyright 2021 NREL

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
import os
import pandas as pd

import floris.tools as wfct
from floris.utilities import wrap_360

from floris_scada_analysis.dataframe_operations import \
    dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as ftls


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
    csv_path = os.path.join(root_dir, 'nwtc_2019_2020_rawdata.csv')

    if not os.path.exists(csv_path):
        print('File not found. Beginning file download with urllib2...')
        url = (
            'https://midcdmz.nrel.gov/apps/plot.pl?site=NWTC&start=20010824&' + 
            'edy=28&emo=9&eyr=2021&year=2019&month=1&day=1&endyear=2020&' + 
            'endmonth=1&endday=1&time=0&inst=25&inst=37&inst=43&inst=55&' +
            'inst=62&type=data&wrlevel=2&preset=0&first=3&math=0&second=-1' +
            '&value=0.0&user=0&axis=1'
            )
        urllib.request.urlretrieve(url, os.path.join(root_dir, csv_path))

    df = pd.read_csv(csv_path)
    return df


if __name__ == '__main__':
    # Determine a wind rose based on met mast data from NWTC
    print("=================================================================")
    print('Downloading and importing met mast data from the NWTC website')
    df = get_wind_data_from_nwtc()

    # Filter to ensure ascending time and no time duplicates
    print("Sorting by time and localizing time entries to MST timezone")
    time_array = df['DATE (MM/DD/YYYY)'] + ' ' + df['MST'] + ':00'
    df['time'] = pd.to_datetime(time_array, format="%m/%d/%Y %H:%M:%S")
    df = df.set_index('time', drop=True).tz_localize(tz="GMT").reset_index(drop=False)

    # Sort dataframe by time and fix duplicates
    df = dfm.df_sort_and_fix_duplicates(df)

    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    fi.vis_layout()

    # Format columns to generic names
    print('Formatting the dataframe with met mast data...')
    df = df.drop(columns=['DATE (MM/DD/YYYY)', 'MST'])
    df = df.rename(
        columns={
            'Avg Wind Speed @ 80m [m/s]': 'ws',
            'Avg Wind Speed (std dev) @ 80m [m/s]': 'ws_std',
            'Avg Wind Direction @ 80m [deg]': 'wd',
            'Avg Wind Direction (std dev) @ 80m [deg]': 'wd_std',
            'Turbulence Intensity @ 80m': 'ti'
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
            wd_array=np.arange(0., 360.0, 3.0),
            ws_array=np.arange(0., 27.0, 1.0),
            ti_array=np.arange(0.03, 0.30, 0.03),
            num_workers=4,
            num_threads=20,
            include_unc=False,
            use_mpi=False
        )
        df_approx.to_feather(fn_approx)

    # Now interpolate solutions from precalculated table to dataframe
    df_fi = ftls.interpolate_floris_from_df_approx(df=df,
        df_approx=df_approx, method='linear', verbose=True)

    # Add noise and bias per turbine
    np.random.seed(123)  # Fixed seed for reproducability
    for ti in range(len(fi.layout_x)):
        df_fi['pow_%03d' % ti] *= 1. + .03 * np.random.randn(df_fi.shape[0])
        df_fi['wd_%03d' % ti] += 3. * np.random.randn(df_fi.shape[0])
        df_fi['ws_%03d' % ti] += 0.2 * np.random.randn(df_fi.shape[0])
        df_fi['ti_%03d' % ti] += 0.01 * np.random.randn(df_fi.shape[0])

        # Saturate power to rated power after adding noise
        df_fi['pow_%03d' % ti] = np.vstack(
            [
                df_fi['pow_%03d' % ti].values,
                np.array([5000.0] * df_fi.shape[0])
            ]
        ).min(axis=0)

        # Mimic northing error
        df_fi['wd_%03d' % ti] += 50. * (np.random.rand() - 0.50)
        df_fi['wd_%03d' % ti] = wrap_360(df_fi['wd_%03d' % ti])

    # Add curtailment to two turbines for January until mid May
    df_fi.loc[
        (df_fi.index < 250000) & (df_fi['pow_002'] > 3150.0), 'pow_002'
    ] = 3150.0
    df_fi.loc[
        (df_fi.index < 250000) & (df_fi['pow_005'] > 3150.0), 'pow_005'
    ] = 3150.0

    # Rename true 'wd', 'ws' and 'ti' channels
    df_fi = df_fi.rename(
        columns={
            'ws': 'ws_truth',
            'wd': 'wd_truth',
            'ti': 'ti_truth',
        }
    )

    fout = os.path.join(root_path, 'demo_dataset_60s.ftr')
    df_fi.to_feather(fout)
    print("Saved dataset to: '" + fout + "'.")

    plt.show()
