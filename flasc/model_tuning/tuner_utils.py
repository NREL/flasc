import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union

from pathlib import Path

from flasc.dataframe_operations import (
    dataframe_filtering as dff,
    dataframe_manipulations as dfm,
)

from floris.tools import FlorisInterface
# from floris.tools import ParallelComputingInterface


def nested_get(dic: Dict[str, Any],
                 keys: List[str]) -> Any:
    """Get a value from a nested dictionary using a list of keys.
    Based on: stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to get the value from.
        keys (List[str]): A list of keys to traverse the dictionary.

    Returns:
        Any: The value at the end of the key traversal.
    """
    for key in keys:
        dic = dic[key]
    return dic

def nested_set(dic: Dict[str, Any], 
                keys: List[str], 
                value: Any, 
                idx: Optional[int] = None) -> None:
    """Set a value in a nested dictionary using a list of keys.
    Based on: stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to set the value in.
        keys (List[str]): A list of keys to traverse the dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): If the value is an list, the index to change.
         Defaults to None.
    """
    dic_in = dic.copy()

    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    if idx is None:
        # Parameter is a scaler, set directly
        dic[keys[-1]] = value
    else:
        # Parameter is a list, need to first get the list, change the values at idx

        # # Get the underlying list
        par_list = nested_get(dic_in, keys)
        par_list[idx] = value
        dic[keys[-1]] = par_list


def set_fi_param(fi_in: FlorisInterface, 
        param: List[str], 
        value: Any, 
        idx: Optional[int] = None) -> FlorisInterface:
    """Set a parameter in a FlorisInterface object.

    Args:
        fi_in (FlorisInterface): The FlorisInterface object to modify.
        param (List[str]): A list of keys to traverse the FlorisInterface dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): The index to set the value at. Defaults to None.

    Returns:
        FlorisInterface: The modified FlorisInterface object.
    """
    fi_dict_mod = fi_in.floris.as_dict()
    nested_set(fi_dict_mod, param, value, idx)
    return FlorisInterface(fi_dict_mod)

# def resim_floris(fi_in: FlorisInterface,
#                  df_scada: pd.DataFrame,
#                  yaw_angles: np.array=None):


#     # Confirm the df_scada has columns 'ws', 'wd'
#     if not all([col in df_scada.columns for col in ['ws', 'wd']]):
#         raise ValueError('df_scada must have columns "ws" and "wd"')
    
#     # Get the number of turbines
#     num_turbines = dfm.get_num_turbines(df_scada)
    
#     # Get wind speeds and directions
#     wind_speeds = df_scada['ws'].values
#     wind_directions = df_scada['wd'].values

#     # Set up the FLORIS model
#     fi = fi_in.copy()
#     fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions, time_series=True)
#     fi.calculate_wake(yaw_angles=yaw_angles)

#     # Get the turbines in kW
#     turbine_powers = fi.get_turbine_powers().squeeze()/1000

#     # Generate FLORIS dataframe
#     df_floris = pd.DataFrame(data=turbine_powers,
#                                 columns=[f'pow_{i:>03}' for i in range(num_turbines)])

#     df_floris = df_floris.assign(ws=wind_speeds,
#                                     wd=wind_directions)#,
#                                     # pow_ref=df_floris[[f"pow_{str(i).zfill(3)}" for i in pow_ref_columns]].mean(axis=1))

#     return df_floris

# def resim_floris_par(fi_in: FlorisInterface,
#                  df_scada: pd.DataFrame,
#                  yaw_angles: np.array=None):



#     # Confirm the df_scada has columns 'ws', 'wd'
#     if not all([col in df_scada.columns for col in ['ws', 'wd']]):
#         raise ValueError('df_scada must have columns "ws" and "wd"')

#     # Get the number of turbines
#     num_turbines = dfm.get_num_turbines(df_scada)

#     max_workers = 16

#     # Set up a parallel computing interface
#     fi = fi_in.copy()
#     fi_pci = ParallelComputingInterface(
#         fi=fi,
#         max_workers=max_workers,
#         n_wind_direction_splits=max_workers,
#         print_timings=True,
#     )

    
#     # Get wind speeds and directions
#     wind_speeds = df_scada['ws'].values
#     wind_directions = df_scada['wd'].values

#     # Set up the FLORIS model
#     fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions, time_series=True)
#     fi.calculate_wake(yaw_angles=yaw_angles)

#     # Get the turbines in kW
#     turbine_powers = fi.get_turbine_powers().squeeze()/1000

#     # Generate FLORIS dataframe
#     df_floris = pd.DataFrame(data=turbine_powers,
#                                 columns=[f'pow_{i:>03}' for i in range(num_turbines)])

#     df_floris = df_floris.assign(ws=wind_speeds,
#                                     wd=wind_directions)#,
#                                     # pow_ref=df_floris[[f"pow_{str(i).zfill(3)}" for i in pow_ref_columns]].mean(axis=1))

#     return df_floris


from flasc.utilities_examples import load_floris_smarteole
if __name__ == "__main__":
    
    fi, _ = load_floris_smarteole(wake_model="emgauss")

    # Testing parameter setting
    # fi_dict_mod = fi.floris.as_dict()

    # param = ['wake','wake_velocity_parameters','empirical_gauss',\
    #             'wake_expansion_rates']
    

    # fi_2 = set_fi_param(fi, param, 7777777, idx=1)

    # print(fi_dict_mod)
    # print('******')
    # print(fi_2.floris.as_dict())

    # Load the SCADA data
    scada_path = Path("../../examples_smarteole/postprocessed/df_scada_data_60s_filtered_and_northing_calibrated.ftr")
    df_scada = pd.read_feather(scada_path)

    # Assign ws/wd and pow_ref
    df_scada = df_scada.assign(ws=df_scada['ws_smarteole'],
                                wd=df_scada['wd_smarteole'],
                                pow_ref=df_scada['pow_ref_smarteole'])

    
    # Resim FLORIS
    # df_floris = resim_floris(fi, df_scada)