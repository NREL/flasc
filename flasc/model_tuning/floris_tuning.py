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


def sweep_velocity_model_parameter_annual_total_wake_losses(
        parameter,
        value_candidates,
        df_scada,
        fi_init
    ):
    
    # Evaluate the scada metric in first step
    aep_scada = # NEED FUNCTION FOR THIS f(df_scada, df_freq)
    # then loop over FLORIS parameters

    # Uses total wake losses

    freq = df_freq # Somehow, use the same df_freq for SCADA aep as FLORIS aep

    # Do we use FLASC to compute AEP? how do we do this exactly?

    aep_floris_list = _sweep_parameter(
        parameter,
        value_candidates,
        fi_init,
        evaluate_aep,
        {"freq":freq}
    )

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


def _sweep_parameter(parameter, value_candidates, fi_init, evaluator, evaluator_kwargs):
    """
    Inputs:
        evaluator -- function handle to be evaluated

    """

    # Generate an fi for each parameter based on fi_init
    # (probably want a special function for this, can get from floris_tuner.py)
    fi_list = [] #< Will contain the fis

    # Make the equivalent df for each fi, put into list
    # Now, we're going to need the ws, wd from df_scada, which is a little 
    # awkward to get? But can be passed in, I guess.
    df_fi_list = []

    values_floris = [evaluator(df_fi, fi, **evaluator_kwargs) for df_fi, fi in zip(df_fi_list, fi_list)]
    
    # Compute the errors (might be a bit more complex than this, but gives an idea)

    return values_floris

def evaluate_aep(df, fi, freq, yaw_angles=None):
    # Possible function to evaluate in _sweep_parameter()

    return fi.get_farm_AEP(freq, yaw_angles=yaw_angles, nowake=False)

def evaluate_energy_ratio(df, fi,):
    # Possible function to evaluate in _sweep_parameter()

    return energy_ratio # what exactly is returned here?


def evaluate_energy_ratio_uplift(df, fi,):
    # Possible function to evaluate in _sweep_parameter()

    # Here, the df will have to have both control modes, presumably

    pass

