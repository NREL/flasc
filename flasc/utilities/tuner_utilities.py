"""Utilities for tuning FLORIS parameters.

This module contains utilities for tuning FLORIS parameters. This includes
functions for resimulating FLORIS with SCADA data, and functions for setting
parameters in a FLORIS model.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from floris import FlorisModel

from flasc import FlascDataFrame
from flasc.data_processing import dataframe_manipulations as dfm
from flasc.utilities.utilities_examples import load_floris_smarteole


def replicate_nan_values(
    df_1: Union[pd.DataFrame, FlascDataFrame],
    df_2: Union[pd.DataFrame, FlascDataFrame],
):
    """Replicate NaN Values in DataFrame df_2 to Match DataFrame df_1.

    For columns that are common between df_1 and df_2, this function ensures that
    NaN values in df_2 appear in the same locations as NaN values in df_1. This is
    primarily useful when df_2 represents a FLORIS resimulation of
    df_1, and you want to ensure that missing data is consistent between the two DataFrames.

    Args:
        df_1 (pandas.DataFrame | FlascDataFrame): The reference DataFrame containing NaN values.
        df_2 (pandas.DataFrame | FlascDataFrame): The DataFrame to be updated
             to match NaN positions in df_1.

    Returns:
        pandas.DataFrame: A new DataFrame with NaN values in df_2 replaced to match df_1.
    """
    # For columns which df_1 and df_2 have in common, make sure
    # occurrences of NaNs which appear in df_1
    # appear in the same location in df_2
    # This function is primarily for the case where df_2 is
    # a FLORIS resimulation of df_1 and making sure
    # missing data appears in both data frames

    # Identify common columns between df_1 and df_2
    common_columns = df_1.columns.intersection(df_2.columns)

    # Remove the time column from the common columns if included
    common_columns = common_columns.drop("time", errors="ignore")

    # Use assign to create a new DataFrame with NaN values replaced
    df_2_updated = df_2.assign(
        **{col: np.where(df_1[col].isna(), np.nan, df_2[col]) for col in common_columns}
    )

    return df_2_updated


def nested_get(dic: Dict[str, Any], keys: List[str]) -> Any:
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


def nested_set(dic: Dict[str, Any], keys: List[str], value: Any, idx: Optional[int] = None) -> None:
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


def resim_floris(
    fm_in: FlorisModel, df_scada: Union[pd.DataFrame, FlascDataFrame], yaw_angles: np.array = None
):
    """Resimulate FLORIS with SCADA data.

    This function takes a FlorisModel  and a SCADA dataframe, and resimulates the
    FlorisModel with the SCADA data. The SCADA data is expected to contain columns
    for wind speed, wind direction, and power reference. The function returns a
    dataframe containing the power output of each turbine in the FlorisModel.

    Args:
        fm_in (FlorisModel): The FlorisModel to resimulate.
        df_scada (pd.DataFrame | FlascDataFrame): The SCADA data to use for resimulation.
        yaw_angles (np.array, optional): The yaw angles to use for resimulation. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the power output of each turbine in the FlorisModel.
    """
    # Get wind speeds and directions
    wind_speeds = df_scada["ws"].values
    wind_directions = df_scada["wd"].values
    # TODO: better handling of TIs?
    turbulence_intensities = fm_in.turbulence_intensities[0] * np.ones_like(wind_speeds)

    # Get the number of turbines
    num_turbines = dfm.get_num_turbines(df_scada)

    # Set up the FLORIS model
    fm = fm_in.copy()
    fm.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        yaw_angles=yaw_angles,
    )
    fm.run()

    # Get the turbines in kW
    turbine_powers = fm.get_turbine_powers().squeeze() / 1000

    # Generate FLORIS dataframe
    df_floris = FlascDataFrame(
        data=turbine_powers, columns=[f"pow_{i:>03}" for i in range(num_turbines)]
    )

    # Assign the FLORIS results to a dataframe
    df_floris = df_floris.assign(ws=wind_speeds, wd=wind_directions)  # ,
    # pow_ref=df_floris[[f"pow_{str(i).zfill(3)}" for i in pow_ref_columns]].mean(axis=1))

    # Make sure the NaN values in the SCADA data appear in the same locations in the
    # FLORIS data
    df_floris = replicate_nan_values(df_scada, df_floris)

    # If df_scada includes a df_mode column copy it over to floris
    if "df_mode" in df_scada.columns:
        df_floris["df_mode"] = df_scada["df_mode"].values

    return df_floris


if __name__ == "__main__":
    fi, _ = load_floris_smarteole(wake_model="emgauss")

    # Testing parameter setting
    # fi_dict_mod = fm.core.as_dict()

    # param = ['wake','wake_velocity_parameters','empirical_gauss',\
    #             'wake_expansion_rates']

    # fi_2 = set_fi_param(fi, param, 7777777, idx=1)

    # print(fi_dict_mod)
    # print('******')
    # print(fi_2.core.as_dict())

    # Load the SCADA data
    scada_path = Path(
        "../../examples_smarteole/postprocessed/df_scada_data_60s_filtered_and_northing_calibrated.ftr"
    )
    df_scada = pd.read_feather(scada_path)

    # Assign ws/wd and pow_ref
    df_scada = df_scada.assign(
        ws=df_scada["ws_smarteole"],
        wd=df_scada["wd_smarteole"],
        pow_ref=df_scada["pow_ref_smarteole"],
    )

    # Resim FLORIS
    # df_floris = resim_floris(fi, df_scada)
