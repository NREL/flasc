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
import pytz

import floris.tools as wfct
from floris_scada_analysis import dataframe_manipulations as dfm
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
    print('Beginning file download with urllib2...')

    url = (
        'https://midcdmz.nrel.gov/apps/plot.pl?site=NWTC&start=2001082' +
        '4&edy=28&emo=2&eyr=2021&year=2019&month=1&day=1&endyear=2020&' +
        'endmonth=1&endday=1&time=0&inst=21&inst=39&inst=58&type=data&' +
        'wrlevel=2&preset=0&first=3&math=0&second=-1&value=0.0&user=0&' +
        'axis=1'
        )

    root_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root_dir, 'tmp_nwtc_2019.csv')
    urllib.request.urlretrieve(url, os.path.join(root_dir, csv_path))

    df = pd.read_csv(csv_path)
    os.remove(csv_path)

    return df


if __name__ == '__main__':
    # Determine a wind rose based on met mast data from NWTC
    print('Downloading and importing met mast data from the NWTC website...')
    df = get_wind_data_from_nwtc()

    # Filter to ensure ascending time and no time duplicates
    time_array = df['DATE (MM/DD/YYYY)'] + ' ' + df['MST'] + ':00'
    time_array = pd.to_datetime(time_array)
    time_array = [t.tz_localize(pytz.timezone('MST')) for t in time_array]
    df['time'] = time_array

    # Sort dataframe by time and fix duplicates
    df = dfm.df_sort_and_fix_duplicates(df)

    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    fi.vis_layout()
    plt.show()

    # Generate local wind direction measurements
    print('Formatting the dataframe with met mast data...')
    df = df.drop(columns=['DATE (MM/DD/YYYY)', 'MST'])
    df = df.rename(columns={'Avg Wind Speed @ 80m [m/s]': 'ws',
                            'Avg Wind Direction @ 80m [deg]': 'wd',
                            'Turbulence Intensity @ 80m': 'ti'})

    # Calculate 'true' solutions
    df_fi = ftls.calc_floris_approx(df, fi,
                                    ws_step=1.0,
                                    wd_step=2.0,
                                    ti_step=0.03)

    # Add noise and bias per turbine
    np.random.seed(123)  # Fixed seed for reproducability
    for ti in range(len(fi.layout_x)):
        df_fi['pow_%03d' % ti] *= 1. + .03 * np.random.randn(df_fi.shape[0])
        df_fi['wd_%03d' % ti] += 3. * np.random.randn(df_fi.shape[0])
        df_fi['ws_%03d' % ti] += 0.2 * np.random.randn(df_fi.shape[0])
        df_fi['ti_%03d' % ti] += 0.01 * np.random.randn(df_fi.shape[0])
        df_fi['wd_%03d' % ti] += 10. * (np.random.rand() - 0.50)  # Bias

    # Drop true 'wd', 'ws' and 'ti' channels
    df_fi = df_fi.drop(columns=['wd', 'ws', 'ti'])

    root_dir = os.path.dirname(os.path.abspath(__file__))
    fout = os.path.join(root_dir, 'demo_dataset_60s.ftr')
    df_fi.to_feather(fout)
    print("Saved dataset to: '" + fout + "'.")
