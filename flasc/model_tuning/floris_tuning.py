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
from flasc.model_tuning.tuner_utils import set_fi_param


def sweep_velocity_model_parameter_for_overall_wake_losses(
        parameter,
        value_candidates,
        df_scada,
        fi_init
    ):
    
    # Evaluate the scada metric in first step
    aep_scada = # NEED FUNCTION FOR THIS f(df_scada, df_freq)
    # then loop over FLORIS parameters

    # Uses total wake losses

    # How should we define df_freq here? as a FLORIS wind rose? 
    freq = df_freq # Somehow, use the same df_freq for SCADA aep as FLORIS aep

    # Do we use FLASC to compute AEP? how do we do this exactly?

    aep_floris_list = _sweep_parameter(
        parameter,
        value_candidates,
        fi_init,
        evaluate_overall_wake_loss,
        {"freq":freq}
    )

    # Also call for SCADA

    aep_floris_nowake = fi_init.get_farm_AEP(freq, nowake=True)

    aep_loss_scada = aep_floris_nowake - aep_scada
    aep_loss_floris = aep_floris_nowake - np.array(aep_floris_list)

    # COULD normalize to a percentage wake loss, if desired... not sure 
    # it's necessary. But would be an option

    # ASSUMING this function should return the error between SCADA and FLORIS:

    return aep_loss_scada - aep_loss_floris

    #OR

    # squared error...
    return (aep_loss_scada - aep_loss_floris)**2

    # I think I prefer an upper-level optimizer to handle whether to square 
    # the error or not.

def sweep_deflection_model_parameter_for_er_uplift():
    
    # Evaluate the scada metric in first step
    # then loop over FLORIS parameters

    # uses energy ratio uplift

    pass

def sweep_wd_std_for_er():

    # uses energy ratio

    pass


def _sweep_floris_parameter(
        parameter,
        value_candidates,
        fi_init,
        evaluator,
        evaluator_kwargs
    ):
    """
    Inputs:
        evaluator -- function handle to be evaluated

    """

    # Generate an fi for each parameter based on fi_init
    if parameter == "wd_std":
        fi_list = [] # TODO: setting wd_std
    else:
        param_idx = None # TODO: handling for param_idx

        # Run FLORIS for each row in df_scada

        # Map over NaNs

        # compute fi overall power
        fi_list = [set_fi_param(fi_init, parameter, v, param_idx) for v in value_candidates]

    values_floris = [evaluator(df_fi, **evaluator_kwargs) for fi in zip(fi_list)]
    
    # Compute the errors (might be a bit more complex than this, but gives an idea)

    return values_floris

def evaluate_overall_wake_loss(df_, freq=None, yaw_angles=None):
    # Possible function to evaluate in _sweep_parameter()



    # Ultimately metric is [ref.sum() - test.sum()] / ref.sum()

    return fi.get_farm_AEP(freq, yaw_angles=yaw_angles, nowake=False) # TODO: replace

def evaluate_energy_ratio(fi,):
    # Possible function to evaluate in _sweep_parameter()

    # Will need to create df_fi here. Should be doable.

    return energy_ratio # what exactly is returned here?


def evaluate_energy_ratio_uplift(df, fi,):
    # Possible function to evaluate in _sweep_parameter()

    # Here, the df will have to have both control modes, presumably

    pass

