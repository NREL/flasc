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


def replicate_nan_values(df_1: pd.DataFrame, 
                         df_2: pd.DataFrame):
    """
    Replicate NaN Values in DataFrame df_2 to Match DataFrame df_1.

    For columns that are common between df_1 and df_2, this function ensures that
    NaN values in df_2 appear in the same locations as NaN values in df_1. This is
    primarily useful when df_2 represents a FLORIS resimulation of
    df_1, and you want to ensure that missing data is consistent between the two DataFrames.

    Parameters:
    - df_1 (pandas.DataFrame): The reference DataFrame containing NaN values.
    - df_2 (pandas.DataFrame): The DataFrame to be updated to match NaN positions in df_1.

    Returns:
    - pandas.DataFrame: A new DataFrame with NaN values in df_2 replaced to match df_1.
    """
    # For columns which df_1 and df_2 have in common, make sure occurences of NaNs which appear in df_1
    # appear in the same location in df_2
    # This function is primarily for the case where df_2 is a FLORIS resimulation of df_1 and making sure
    # missing data appears in both data frames

    # Identify common columns between df_1 and df_2
    common_columns = df_1.columns.intersection(df_2.columns)

    # Use assign to create a new DataFrame with NaN values replaced
    df_2_updated = df_2.assign(**{col: np.where(df_1[col].isna(), np.nan, df_2[col]) for col in common_columns})

    return df_2_updated    

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
        param_idx: Optional[int] = None) -> FlorisInterface:
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
    nested_set(fi_dict_mod, param, value, param_idx)
    return FlorisInterface(fi_dict_mod)


def resim_floris(fi_in: FlorisInterface,
                  df_scada: pd.DataFrame,
                  yaw_angles: np.array=None):

        # Get wind speeds and directions
        wind_speeds = df_scada['ws'].values
        wind_directions = df_scada['wd'].values

        # Get the number of turbiens
        num_turbines = dfm.get_num_turbines(df_scada)

        # Set up the FLORIS model
        fi = fi_in.copy()
        fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions, time_series=True)
        fi.calculate_wake(yaw_angles=yaw_angles)
        
        # Get the turbines in kW
        turbine_powers = fi.get_turbine_powers().squeeze()/1000

        # Generate FLORIS dataframe
        df_floris = pd.DataFrame(data=turbine_powers,
                                    columns=[f'pow_{i:>03}' for i in range(num_turbines)])

        # Assign the FLORIS results to a dataframe
        df_floris = df_floris.assign(ws=wind_speeds,
                                        wd=wind_directions)#,
                                        # pow_ref=df_floris[[f"pow_{str(i).zfill(3)}" for i in pow_ref_columns]].mean(axis=1))

        # Make sure the NaN values in the SCADA data appear in the same locations in the
        # FLORIS data
        df_floris = replicate_nan_values(df_scada, df_floris)

        # If df_scada includes a df_mode column copy it over to floris
        if 'df_mode' in df_scada.columns:
            df_floris['df_mode'] = df_scada['df_mode'].values

        return df_floris

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
