# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

# The purpose of these tests is to provide a consistent timing test function
# to the flasc_metrics repository.  Even if the FLASC API changes internally
# these functions should perform equivalent tasks and provide a consistent
# timing test.

import os
import warnings
import time

import numpy as np
import pandas as pd

from flasc.energy_ratio import energy_ratio_suite

N_ITERATIONS = 5


def load_data_and_prep_data():
    # Load dataframe with artificial SCADA data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir, '..','examples_artificial_data', 'raw_data_processing', 'postprocessed', 'df_scada_data_600s_filtered_and_northing_calibrated.ftr'
    )

    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            'Please run the scripts in /raw_data_processing/' +
            'before trying any of the other examples.'
        )
    
    df = pd.read_feather(ftr_path)
        
    # Let 0 be the reference turbine (pow/ws/wd) and 1 be the test turbine
    df['ws'] = df['ws_000']
    df['wd'] = df['wd_000']
    df['pow_ref'] = df['pow_000']
    df['pow_test'] = df['pow_001']

    return df

# Time how long it takes to compute the energy ratio for a single turbine
# using N=20 bootstraps
def time_energy_ratio_with_bootstrapping():

    # Number of bootstraps
    N = 20

    # Load the data
    df = load_data_and_prep_data()

    # Load an energy ratio suite from FLASC
    s = energy_ratio_suite.energy_ratio_suite(verbose=False)

    # Add dataframe to energy suite
    s.add_df(df, 'data')

    # For forward consistency, define the bins by the edges
    ws_edges = np.arange(5,25,1.)
    wd_edges = np.arange(0,360,2.)

    # Create bins
    ws_bins = [(ws_edges[i], ws_edges[i+1]) for i in range(len(ws_edges)-1)]
    wd_bins = [(wd_edges[i], wd_edges[i+1]) for i in range(len(wd_edges)-1)]

    # Run this calculation N_ITERATIONS times and take the average time
    
    time_results = np.zeros(N_ITERATIONS)
    for i in range(N_ITERATIONS):
        start_time = time.time()
        er = s.get_energy_ratios(
            test_turbines=1,
            ws_bins=ws_bins,
            wd_bins=wd_bins,
            N=N,
            percentiles=[5.0, 95.0],
            verbose=False
        )

        end_time = time.time()
        time_results[i] = end_time - start_time

    # Return the average time
    return np.mean(time_results)





if __name__=="__main__":
    warnings.filterwarnings('ignore')

    # Test loading the data
    df = load_data_and_prep_data()
    print(df.head())
    print(df.shape)

    # Test timing the energy ratio
    print(time_energy_ratio_with_bootstrapping())