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
from flasc.model_tuning.tuner_utils import set_fi_param, resim_floris

from flasc.dataframe_operations import (
    dataframe_filtering as dff,
    dataframe_manipulations as dfm,
)



from floris.tools import FlorisInterface


def evaluate_overall_wake_loss(df_,
                               df_freq=None):
    # Evaluate the overall deficit in energy from the column 'ref_pow'
    # to the column test_pow

    # Not sure yet if we want to figure out how to use df_freq here
    return 100 * (df_['pow_ref'].sum()  - df_['pow_test'].sum()) / df_['pow_ref'].sum()


def sweep_velocity_model_parameter_for_overall_wake_losses(
        parameter,
        value_candidates,
        df_scada,
        fi_in,
        param_idx = None,
        yaw_angles = None,
        df_freq = None # Not yet certain we will use this
    ):

    # Currently assuming pow_ref and pow_test already assigned
    # Also assuming limit to ws/wd range accomplished but could revisit?
    
    # First collect the scada wake loss
    scada_wake_loss = evaluate_overall_wake_loss(df_scada)

    # Now loop over FLORIS candidates and collect the wake loss
    floris_wake_losses = np.array(len(value_candidates))
    for idx, vc in enumerate(value_candidates):
        
        # Set the parameter
        fi = set_fi_param(fi_in, parameter, vc, param_idx)

        # Collect the FLORIS results
        df_floris = resim_floris(fi, df_scada, yaw_angles=yaw_angles)

        # Get the wake loss
        floris_wake_losses[idx] = evaluate_overall_wake_loss(df_floris)

    # Return the error
    return floris_wake_losses - scada_wake_loss





    # Uses total wake losses



    # # Do we use FLASC to compute AEP? how do we do this exactly?
    # aep_floris_list = _sweep_parameter(
    #     parameter,
    #     value_candidates,
    #     fi_init,
    #     evaluate_overall_wake_loss,
    #     {"freq":freq}
    # )

    # # Also call for SCADA

    # aep_floris_nowake = fi_init.get_farm_AEP(freq, nowake=True)

    # aep_loss_scada = aep_floris_nowake - aep_scada
    # aep_loss_floris = aep_floris_nowake - np.array(aep_floris_list)

    # # COULD normalize to a percentage wake loss, if desired... not sure 
    # # it's necessary. But would be an option

    # # ASSUMING this function should return the error between SCADA and FLORIS:

    # return aep_loss_scada - aep_loss_floris

    # #OR

    # # squared error...
    # return (aep_loss_scada - aep_loss_floris)**2

    # # I think I prefer an upper-level optimizer to handle whether to square 
    # # the error or not.

def sweep_deflection_model_parameter_for_er_uplift():
    
    # Evaluate the scada metric in first step
    # then loop over FLORIS parameters

    # uses energy ratio uplift

    pass

def sweep_wd_std_for_er():

    # uses energy ratio

    pass


# def _sweep_floris_parameter(
#         parameter,
#         value_candidates,
#         fi_init,
#         evaluator,
#         evaluator_kwargs
#     ):
#     """
#     Inputs:
#         evaluator -- function handle to be evaluated

#     """

#     # Generate an fi for each parameter based on fi_init
#     if parameter == "wd_std":
#         fi_list = [] # TODO: setting wd_std
#     else:
#         param_idx = None # TODO: handling for param_idx

#         # Run FLORIS for each row in df_scada

#         # Map over NaNs

#         # compute fi overall power
#         fi_list = [set_fi_param(fi_init, parameter, v, param_idx) for v in value_candidates]

#     values_floris = [evaluator(df_fi, **evaluator_kwargs) for fi in zip(fi_list)]
    
#     # Compute the errors (might be a bit more complex than this, but gives an idea)

#     return values_floris



# def evaluate_energy_ratio(fi,):
#     # Possible function to evaluate in _sweep_parameter()

#     # Will need to create df_fi here. Should be doable.

#     return energy_ratio # what exactly is returned here?


def evaluate_energy_ratio_uplift(df, fi,):
    # Possible function to evaluate in _sweep_parameter()

    # Here, the df will have to have both control modes, presumably

    pass

