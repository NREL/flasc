# Copyright 2023 NREL
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation
import numpy as np
import pandas as pd
import polars as pl
from flasc.model_tuning.tuner_utils import set_fi_param, resim_floris

from flasc.dataframe_operations import (
    dataframe_filtering as dff,
    dataframe_manipulations as dfm,
)

import flasc.floris_tools as ftools
from flasc.energy_ratio.energy_ratio_utilities import add_power_ref, add_power_test
from flasc.energy_ratio.energy_ratio_input import EnergyRatioInput
from flasc.energy_ratio import energy_ratio as er
from sklearn.metrics import mean_squared_error
from flasc.model_tuning.tuner_utils import replicate_nan_values

from floris.tools import FlorisInterface, UncertaintyInterface

from flasc.energy_ratio import total_uplift as tup

def evaluate_overall_wake_loss(df_,
                               df_freq=None):
    # Evaluate the overall deficit in energy from the column 'ref_pow'
    # to the column test_pow

    # Not sure yet if we want to figure out how to use df_freq here
    return 100 * (df_['pow_ref'].sum()  - df_['pow_test'].sum()) / df_['pow_ref'].sum()


def sweep_velocity_model_parameter_for_overall_wake_losses(
        parameter,
        value_candidates,
        df_scada_in,
        fi_in,
        ref_turbines,
        test_turbines,
        param_idx = None,
        yaw_angles = None,
        wd_min = 0.0,
        wd_max = 360.0,
        # ws_step: float = 1.0,
        ws_min = 0.0,
        ws_max = 50.0,
        df_freq = None # Not yet certain we will use this
    ):

    # Currently assuming pow_ref and pow_test already assigned
    # Also assuming limit to ws/wd range accomplished but could revisit?

    # Assign the ref and test cols
    df_scada = pl.from_pandas(df_scada_in)

    # Trim to ws/wd
    df_scada = df_scada.filter(
        (pl.col('ws') >= ws_min) &  # Filter the mean wind speed
        (pl.col('ws') < ws_max) &
        (pl.col('wd') >= wd_min) &  # Filter the mean wind direction
        (pl.col('wd') < wd_max) 
    )

    ref_cols = [f'pow_{i:03d}' for i in ref_turbines]
    test_cols = [f'pow_{i:03d}' for i in test_turbines]
    df_scada = add_power_ref(df_scada, ref_cols)
    df_scada = add_power_test(df_scada, test_cols)
    
    # First collect the scada wake loss
    scada_wake_loss = evaluate_overall_wake_loss(df_scada)

    # Now loop over FLORIS candidates and collect the wake loss
    floris_wake_losses = np.zeros(len(value_candidates))
    for idx, vc in enumerate(value_candidates):
        
        # Set the parameter
        fi = set_fi_param(fi_in, parameter, vc, param_idx)

        # Collect the FLORIS results
        df_floris = resim_floris(fi, df_scada.to_pandas(), yaw_angles=yaw_angles)
        df_floris = pl.from_pandas(df_floris)

        # Assign the ref and test cols
        df_floris = add_power_ref(df_floris, ref_cols)
        df_floris = add_power_test(df_floris, test_cols)

        # Get the wake loss
        floris_wake_losses[idx] = evaluate_overall_wake_loss(df_floris)

    # Return the error
    return floris_wake_losses, scada_wake_loss

def select_best_velocity_parameter(floris_reults, 
                       scada_results,
                       value_candidates,
                       ax=None):
    
    error_values = (floris_reults - scada_results)**2

    best_param = value_candidates[np.argmin(error_values)]
    best_floris_result = floris_reults[np.argmin(error_values)]

    if ax is not None:

        ax.plot(value_candidates, floris_reults, 'b.-', label='FLORIS')
        ax.scatter(best_param,best_floris_result,color='r',marker='o', label='Best Fit')
        ax.axhline(scada_results,color='k', label='SCADA')
        ax.grid(True)
        ax.legend()

    return best_param

def sweep_wd_std_for_er(
        value_candidates,
        df_scada_in,
        df_approx_,
        ref_turbines,
        test_turbines,
        yaw_angles = None,
        wd_step = 2.0,
        wd_min = 0.0,
        wd_max = 360.0,
        ws_step: float = 1.0,
        ws_min = 0.0,
        ws_max = 50.0,
        bin_cols_in = ['wd_bin','ws_bin'],
        weight_by = 'min', #min, sum
        df_freq = None, # Not yet certain we will use this,
        remove_all_nulls = False
    ):

    # Currently assuming pow_ref and pow_test already assigned
    # Also assuming limit to ws/wd range accomplished but could revisit?

    # Assign the ref and test cols
    df_scada = pl.from_pandas(df_scada_in)
    

    # Trim to ws/wd
    df_scada = df_scada.filter(
        (pl.col('ws') >= ws_min) &  # Filter the mean wind speed
        (pl.col('ws') < ws_max) &
        (pl.col('wd') >= wd_min) &  # Filter the mean wind direction
        (pl.col('wd') < wd_max) 
    )

    ref_cols = [f'pow_{i:03d}' for i in ref_turbines]
    test_cols = [f'pow_{i:03d}' for i in test_turbines]
    df_scada = add_power_ref(df_scada, ref_cols)
    df_scada = add_power_test(df_scada, test_cols)
    
    df_scada = df_scada.to_pandas()
    df_scada['ti'] = 0.1

    
    # scada_vals = er_out.df_result['SCADA'].values
    
    # # First collect the scada wake loss
    # scada_wake_loss = evaluate_overall_wake_loss(df_scada)

    # Now loop over FLORIS candidates and collect the wake loss
    er_error = np.zeros(len(value_candidates))
    df_list = []
    for idx, wd_std in enumerate(value_candidates):
        
        if wd_std > 0:
            df_approx_wd_std = ftools.add_gaussian_blending_to_floris_approx_table(df_approx_, wd_std)
        else:
            df_approx_wd_std = df_approx_.copy()

        df_floris = ftools.interpolate_floris_from_df_approx(df_scada,
                                                             df_approx_wd_std,
                                                             mirror_nans=False,
                                                             wrap_0deg_to_360deg=False)
        df_floris = replicate_nan_values(df_scada,df_floris)

        # Collect the FLORIS results
        # df_floris = resim_floris(fi, df_scada.to_pandas(), yaw_angles=yaw_angles)
        df_floris = pl.from_pandas(df_floris)

        # Assign the ref and test cols
        df_floris = add_power_ref(df_floris, ref_cols)
        df_floris = add_power_test(df_floris, test_cols)

        # Compare the energy ratio to SCADA
        er_in = EnergyRatioInput(
            [df_scada, df_floris.to_pandas()], 
            ["SCADA", "FLORIS"]
        )

        er_out = er.compute_energy_ratio(
            er_in,
            ref_turbines=ref_turbines,
            test_turbines=test_turbines,
            # use_predefined_ref=use_predefined_ref,
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            N=1,
        )

        df_ = (er_out.df_result
               .copy()
        )

        df_list.append(df_)

        # Grab the energy ratios and counts
        scada_vals = df_['SCADA'].values
        floris_vals = df_['FLORIS'].values
        count_vals = df_['count_SCADA'].values

        er_error[idx] = mean_squared_error(y_true=scada_vals, 
                                 y_pred=floris_vals, 
                                 sample_weight=count_vals)
        
    # Return the error
    return er_error, df_list


def select_best_wd_std(er_results, 
                       value_candidates,
                       ax=None):
    
    error_sq = er_results**2

    best_param = value_candidates[np.argmin(error_sq)]
    

    if ax is not None:

        ax.plot(value_candidates, error_sq, 'b.-', label='Energy Ratio Error')
        ax.axvline(best_param,color='r')
        ax.set_xlabel('wd_std')
        ax.set_ylabel('squared error')
        ax.grid(True)
        ax.legend()

    return best_param


def sweep_deflection_parameter_for_total_uplift(
        parameter,
        value_candidates,
        df_scada_baseline_in,
        df_scada_wakesteering_in,
        fi_in,
        ref_turbines,
        test_turbines,
        yaw_angles_baseline = None, 
        yaw_angles_wakesteering = None, 
        wd_step = 2.0,
        wd_min = 0.0,
        wd_max = 360.0,
        ws_step: float = 1.0,
        ws_min = 0.0,
        ws_max = 50.0,
        bin_cols_in = ['wd_bin','ws_bin'],
        weight_by = 'min', #min, sum
        df_freq = None, # Not yet certain we will use this,
        remove_all_nulls = False
    ):

    # Currently assuming pow_ref and pow_test already assigned
    # Also assuming limit to ws/wd range accomplished but could revisit?

    # Assign the ref and test cols
    df_scada_baseline = pl.from_pandas(df_scada_baseline_in)
    df_scada_wakesteering = pl.from_pandas(df_scada_wakesteering_in)
    

    # Trim to ws/wd
    df_scada_baseline = df_scada_baseline.filter(
        (pl.col('ws') >= ws_min) &  # Filter the mean wind speed
        (pl.col('ws') < ws_max) &
        (pl.col('wd') >= wd_min) &  # Filter the mean wind direction
        (pl.col('wd') < wd_max) 
    )
    df_scada_wakesteering = df_scada_wakesteering.filter(
        (pl.col('ws') >= ws_min) &  # Filter the mean wind speed
        (pl.col('ws') < ws_max) &
        (pl.col('wd') >= wd_min) &  # Filter the mean wind direction
        (pl.col('wd') < wd_max) 
    )

    ref_cols = [f'pow_{i:03d}' for i in ref_turbines]
    test_cols = [f'pow_{i:03d}' for i in test_turbines]
    df_scada_baseline = add_power_ref(df_scada_baseline, ref_cols)
    df_scada_baseline = add_power_test(df_scada_baseline, test_cols)
    df_scada_wakesteering = add_power_ref(df_scada_wakesteering, ref_cols)
    df_scada_wakesteering = add_power_test(df_scada_wakesteering, test_cols)
    
    df_scada_baseline = df_scada_baseline.to_pandas()
    df_scada_wakesteering = df_scada_wakesteering.to_pandas()

    # Compare the scada uplift
    er_in = EnergyRatioInput(
        [df_scada_baseline, df_scada_wakesteering], 
        ["Baseline [SCADA]", "Controlled [SCADA]"]
    )

    scada_uplift_result = tup.compute_total_uplift(
        er_in,
        test_turbines=test_turbines,
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=wd_step,
        wd_min=wd_min,
        wd_max=wd_max,
        ws_step=ws_step,
        ws_min=ws_min,
        ws_max=ws_max,
        uplift_pairs=[("Baseline [SCADA]", "Controlled [SCADA]")],
        uplift_names=["Uplift [SCADA]"],
        N=1,
    )

    print(scada_uplift_result)
    scada_uplift = scada_uplift_result["Uplift [SCADA]"]["energy_uplift_ctr_pc"]
    print(scada_uplift)

    
    # Now loop over FLORIS candidates and collect the uplift
    floris_uplifts = np.zeros(len(value_candidates))
    # df_list = []
    for idx, vc in enumerate(value_candidates):

         # Set the parameter for baseline and wake steering
        fi_baseline = set_fi_param(fi_in, parameter, vc)
        fi_wakesteering = fi_baseline.copy()

        # Collect the FLORIS results
        df_floris_baseline = resim_floris(fi_baseline, df_scada_baseline, yaw_angles=yaw_angles_baseline)
        df_floris_wakesteering = resim_floris(fi_wakesteering, df_scada_wakesteering, yaw_angles=yaw_angles_wakesteering)

        df_floris_baseline = pl.from_pandas(df_floris_baseline)
        df_floris_wakesteering = pl.from_pandas(df_floris_wakesteering)

        # Assign the ref and test cols
        df_floris_baseline = add_power_ref(df_floris_baseline, ref_cols)
        df_floris_baseline = add_power_test(df_floris_baseline, test_cols)
        df_floris_wakesteering = add_power_ref(df_floris_wakesteering, ref_cols)
        df_floris_wakesteering = add_power_test(df_floris_wakesteering, test_cols)

        # Calculate the FLORIS uplift
        er_in = EnergyRatioInput(
            [df_floris_baseline.to_pandas(), df_floris_wakesteering.to_pandas()], 
            ["Baseline [FLORIS]", "Controlled [FLORIS]"]
        )

        scada_uplift_result = tup.compute_total_uplift(
            er_in,
            test_turbines=test_turbines,
            use_predefined_ref=True,
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_step=wd_step,
            wd_min=wd_min,
            wd_max=wd_max,
            ws_step=ws_step,
            ws_min=ws_min,
            ws_max=ws_max,
            uplift_pairs=[("Baseline [FLORIS]", "Controlled [FLORIS]")],
            uplift_names=["Uplift [FLORIS]"],
            N=1,
        )

        floris_uplifts[idx] = scada_uplift_result["Uplift [FLORIS]"]["energy_uplift_ctr_pc"]

    return floris_uplifts, scada_uplift


        