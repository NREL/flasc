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

from flasc.visualization import plot_layout_with_waking_directions, plot_binned_mean_and_ci



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
    num_points_per_combination = 5 # 5 # How many "seconds" per combination

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

    # Use the function plot_binned_mean_and_ci to show the noise in wind speed
    fig, ax = plt.subplots(1,1,sharex=True)
    plot_binned_mean_and_ci(df_baseline.ws, df_baseline_noisy.ws, ax=ax)
    ax.set_xlabel('Wind Speed (m/s) [Baseline]')
    ax.set_ylabel('Wind Speed (m/s) [Baseline (Noisy)]')
    ax.grid(True)
    


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

    # Quick and dirty test of my idea for overall gain

    # Calculate actual energy gain
    total_energy_baseline = df_baseline['pow_002'].sum()
    total_energy_wakesteering = df_wakesteering['pow_002'].sum()
    total_energy_baseline_noisy = df_baseline_noisy['pow_002'].sum()
    total_energy_wakesteering_noisy = df_wakesteering_noisy['pow_002'].sum()

    # print(fsc.df_list_gains[0]['er_results'])

    print('~~~~~ Assess ability to estimate total energy uplift')
    print('In non-noisy case, energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline, total_energy_wakesteering, 100 * (total_energy_wakesteering -total_energy_baseline )/total_energy_baseline))
    print('In noisy case, energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline_noisy, total_energy_wakesteering_noisy, 100 * (total_energy_wakesteering_noisy -total_energy_baseline_noisy )/total_energy_baseline_noisy))
    
    # Add a bin wd to match up with energy ratio results
    df_baseline['wd_bin'] = np.round( (df_baseline['wd'] - 1) / 2.0) * 2.0 + 1
    df_baseline_noisy['wd_bin'] = np.round( (df_baseline_noisy['wd'] - 1) / 2.0) * 2.0 + 1
   
    # Now produce approximate uplift per bin
    df_merge = (df_baseline
        .groupby('wd_bin')  #Group original results by wind direction
        .sum()  # Sum within bin
        .reset_index()
        .merge(fsc.df_list_gains[0]['er_results'], on='wd_bin') # Combine energy ratio gains
        .assign(
            pred = lambda df_: df_.pow_002 * df_.baseline,  # Assign channels combing baseline energy with gain
            pred_lb = lambda df_: df_.pow_002 * df_.baseline_lb,
            pred_ub = lambda df_: df_.pow_002 * df_.baseline_ub
        )
        .dropna()
    )

    total_energy_baseline = df_merge['pow_002'].sum()
    total_energy_wakesteering = df_merge['pred'].sum()
    total_energy_wakesteering_lb = df_merge['pred_lb'].sum()
    total_energy_wakesteering_ub = df_merge['pred_ub'].sum()

    print('====== Predictions (NON-NOISY) =====')
    print('Uplift from gain now for NON-NOISY Case, based on energy ratio gain')
    print('In non-noisy case (lower bound), energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline, total_energy_wakesteering_lb, 100 * (total_energy_wakesteering_lb -total_energy_baseline )/total_energy_baseline))
    print('In non-noisy case (central), energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline, total_energy_wakesteering, 100 * (total_energy_wakesteering -total_energy_baseline )/total_energy_baseline))
    print('In non-noisy case (upper bound), energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline, total_energy_wakesteering_ub, 100 * (total_energy_wakesteering_ub -total_energy_baseline )/total_energy_baseline))

    # Repeat with noisy data

    df_merge = (df_baseline_noisy
        .groupby('wd_bin')  #Group original results by wind direction
        .sum()  # Sum within bin
        .reset_index()
        .merge(fsc.df_list_gains[1]['er_results'], on='wd_bin') # Combine energy ratio gains
        .assign(
            pred = lambda df_: df_.pow_002 * df_.baseline,  # Assign channels combing baseline energy with gain
            pred_lb = lambda df_: df_.pow_002 * df_.baseline_lb,
            pred_ub = lambda df_: df_.pow_002 * df_.baseline_ub
        )
        .dropna()
    )

    total_energy_baseline = df_merge['pow_002'].sum()
    total_energy_wakesteering = df_merge['pred'].sum()
    total_energy_wakesteering_lb = df_merge['pred_lb'].sum()
    total_energy_wakesteering_ub = df_merge['pred_ub'].sum()

    print('====== Predictions (NOISY) =====')
    print('Uplift from gain now for NOISY Case, based on energy ratio gain')
    print('In NOISY case (lower bound), energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline, total_energy_wakesteering_lb, 100 * (total_energy_wakesteering_lb -total_energy_baseline )/total_energy_baseline))
    print('In NOISY case (central), energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline, total_energy_wakesteering, 100 * (total_energy_wakesteering -total_energy_baseline )/total_energy_baseline))
    print('In NOISY case (upper bound), energy production in total rises from %.1f to %.1f (%.1f%%)' % (total_energy_baseline, total_energy_wakesteering_ub, 100 * (total_energy_wakesteering_ub -total_energy_baseline )/total_energy_baseline))


    plt.show()


