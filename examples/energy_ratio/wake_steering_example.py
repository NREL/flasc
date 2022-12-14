# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from floris import tools as wfct
from floris.utilities import wrap_360

from flasc.energy_ratio import energy_ratio_suite
# from flasc import floris_tools as fsatools

from flasc.visualization import plot_layout_with_waking_directions


if __name__ == "__main__":

    # Construct a simple 3-turbine wind farm with a 
    # Reference turbine (0)
    # Controlled turbine (1)
    # Downstream turbine (2)
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.yaml")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    fi.reinitialize(layout_x = [0, 0, 5*126], layout_y = [5*126, 0, 0])

    # Show the wind farm
    plot_layout_with_waking_directions(fi)

    # Create a time history of points where the wind speed and wind direction step different combinations
    ws_points = np.arange(5.0,10.0,1.0)
    wd_points = np.arange(250.0, 290.0, 1.,)
    num_points_per_combination = 5 # How many "seconds" per combination

    # I know this is dumb but will come back, can't quite work out the numpy version
    ws_array = []
    wd_array = []
    for ws in ws_points:
        for wd in wd_points:
            for i in range(num_points_per_combination):
                ws_array.append(ws)
                wd_array.append(wd)
    t = np.arange(len(ws_array))

    fig, axarr = plt.subplots(2,1,sharex=True)
    axarr[0].plot(t, ws_array,label='Wind Speed')
    axarr[0].set_ylabel('m/s')
    axarr[0].legend()
    axarr[0].grid(True)
    axarr[1].plot(t, wd_array,label='Wind Direction')
    axarr[1].set_ylabel('deg')
    axarr[1].legend()
    axarr[1].grid(True)


    # Compute the power of the second turbine for two cases
    # Baseline: The front turbine is aligned to the wind
    # WakeSteering: The front turbine is yawed 25 deg
    fi.reinitialize(wind_speeds=ws_array, wind_directions=wd_array, time_series=True)
    fi.calculate_wake()
    power_baseline_ref = fi.get_turbine_powers().squeeze()[:,0].flatten()
    power_baseline_control = fi.get_turbine_powers().squeeze()[:,1].flatten()
    power_baseline_downstream = fi.get_turbine_powers().squeeze()[:,2].flatten()

    yaw_angles = np.zeros([len(t),1,3]) * 25
    yaw_angles[:,:,1] = 25 # Set control turbine yaw angles to 25 deg
    fi.calculate_wake(yaw_angles=yaw_angles)
    power_wakesteering_ref = fi.get_turbine_powers().squeeze()[:,0].flatten()
    power_wakesteering_control = fi.get_turbine_powers().squeeze()[:,1].flatten()
    power_wakesteering_downstream = fi.get_turbine_powers().squeeze()[:,2].flatten()

    # Build up the data frames needed for energy ratio suite
    df_baseline = pd.DataFrame({
        'wd':wd_array,
        'ws':ws_array,
        'pow_ref':power_baseline_ref,
        'pow_000':power_baseline_ref, 
        'pow_001':power_baseline_control,
        'pow_002':power_baseline_downstream
    })

    df_wakesteering = pd.DataFrame({
        'wd':wd_array,
        'ws':ws_array,
        'pow_ref':power_wakesteering_ref,
        'pow_000':power_wakesteering_ref, 
        'pow_001':power_wakesteering_control,
        'pow_002':power_wakesteering_downstream
    })

    # Create alternative versions of each of the above dataframes where the wd/ws are perturbed by noise
    df_baseline_noisy = pd.DataFrame({
        'wd':wd_array + np.random.randn(len(wd_array))*5,
        'ws':ws_array + np.random.randn(len(ws_array)),
        'pow_ref':power_baseline_ref,
        'pow_000':power_baseline_ref, 
        'pow_001':power_baseline_control,
        'pow_002':power_baseline_downstream
    })

    df_wakesteering_noisy = pd.DataFrame({
        'wd':wd_array + np.random.randn(len(wd_array))*5,
        'ws':ws_array + np.random.randn(len(ws_array)),
        'pow_ref':power_wakesteering_ref,
        'pow_000':power_wakesteering_ref, 
        'pow_001':power_wakesteering_control,
        'pow_002':power_wakesteering_downstream
    })

    # Make a color palette that visually links the nominal and noisy data sets together
    color_palette = sns.color_palette("Paired",4)[::-1]
    # color_palette = ['r','g','b','k']

    # Initialize the energy ratio suite object and add each dataframe
    # separately. 
    fsc = energy_ratio_suite.energy_ratio_suite()
    fsc.add_df(df_baseline, 'Baseline', color_palette[0])
    fsc.add_df(df_wakesteering, 'WakeSteering', color_palette[1])
    fsc.add_df(df_baseline_noisy, 'Baseline (Noisy)', color_palette[2])
    fsc.add_df(df_wakesteering_noisy, 'WakeSteering (Noisy)', color_palette[3])

    # Print out the energy ratio
    fsc.print_dfs()


    # Calculate and plot the energy ratio for the downstream turbine [2]
    # With respect to reference turbine [0]
    # datasets with uncertainty quantification using 50 bootstrap samples
    fsc.get_energy_ratios(
        test_turbines=[2],
        wd_step=2.0,
        ws_step=1.0,
        N=10,
        percentiles=[5., 95.],
        verbose=False
    )
    fsc.plot_energy_ratios(superimpose=True)

    fsc.get_energy_ratios_gain(
        test_turbines=[2],
        wd_step=2.0,
        ws_step=1.0,
        N=10,
        percentiles=[5., 95.],
        verbose=False
    )
    fsc.plot_energy_ratio_gains(superimpose=True)

    plt.show()

