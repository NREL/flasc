# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
import copy

from floris.tools import FlorisInterface
from floris.utilities import wrap_360
from flasc import utilities as fsut


def merge_floris_objects(fi_list, reference_wind_height=None):
    """Merge a list of FlorisInterface objects into a single FlorisInterface object. Note that it uses
    the very first object specified in fi_list to build upon, so it uses those wake model parameters,
    air density, and so on.

    Args:
        fi_list (list): Array-like of FlorisInterface objects.
        reference_wind_height (float, optional): Height in meters at which the reference wind speed is
        assigned. If None, will assume this value is equal to the reference wind height specified in
        the FlorisInterface objects. This only works if all objects have the same value for their
        reference_wind_height.

    Returns:
        fi_merged (FlorisInterface): The merged FlorisInterface object, merged in the same order as fi_list.
        The objects are merged on the turbine locations and turbine types, but not on the wake parameters
        or general solver settings.
    """

    # Make sure the entries in fi_list are FlorisInterface objects
    if not isinstance(fi_list[0], FlorisInterface):
        raise UserWarning("Incompatible input specified. Please merge FlorisInterface objects before inserting them into ParallelComputingInterface and UncertaintyInterface.")

    # Get the turbine locations and specifications for each subset and save as a list
    x_list = []
    y_list = []
    turbine_type_list = []
    reference_wind_heights = []
    for fi in fi_list:
        x_list.extend(fi.layout_x)
        y_list.extend(fi.layout_y)

        fi_turbine_type = fi.floris.farm.turbine_type
        if len(fi_turbine_type) == 1:
            fi_turbine_type = fi_turbine_type * len(fi.layout_x)
        elif not len(fi_turbine_type) == len(fi.layout_x):
            raise UserWarning("Incompatible format of turbine_type in FlorisInterface.")

        turbine_type_list.extend(fi_turbine_type)
        reference_wind_heights.append(fi.floris.flow_field.reference_wind_height)

    # Derive reference wind height, if unspecified by the user
    if reference_wind_height is None:
        reference_wind_height = np.mean(reference_wind_heights)
        if np.any(np.abs(np.array(reference_wind_heights) - reference_wind_height) > 1.0e-3):
            raise UserWarning("Cannot automatically derive a fitting reference_wind_height since they substantially differ between FlorisInterface objects. Please specify 'reference_wind_height' manually.")

    # Construct the merged FLORIS model based on the first entry in fi_list
    fi_merged = fi_list[0].copy()
    fi_merged.reinitialize(
        layout_x=x_list,
        layout_y=y_list,
        turbine_type=turbine_type_list,
        reference_wind_height=reference_wind_height
    )

    return fi_merged


def interpolate_floris_from_df_approx(
    df,
    df_approx,
    method='linear',
    wrap_0deg_to_360deg=True,
    extrapolate_ws=True,
    extrapolate_ti=True,
    mirror_nans=True,
    verbose=True
):
    """This function generates the FLORIS predictions for a set of historical
    data, 'df', quickly by linearly interpolating from a precalculated set of
    FLORIS solutions, 'df_approx'. We use linear interpolation to eliminate
    dependency of the computation time on the size of the dataframe/number of
    timeseries samples.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the timeseries for
        which the FLORIS predictions should be calculated. It should contain
        at least the columns 'wd', 'ws', and 'ti', which are respectively the
        ambient wind direction, ambient wind speed, and ambient turbulence
        intensity to be used in the FLORIS predictions. An example:

        df=
                 time                    wd     ws      ti  
        0       2018-01-01 00:10:00   213.1   7.81    0.08
        1       2018-01-01 00:20:00   215.6   7.65    0.08
        ...                     ...   ...      ...     ...
        52103   2018-12-31 23:30:00    15.6   11.0    0.08
        52104   2018-12-31 23:40:00    15.3   11.1    0.08

        df_approx (pd.DataFrame): A Pandas DataFrame containing the precalculated
        solutions of the FLORIS model for a large grid of ambient wind directions,
        wind speeds and turbulence intensities. This table is typically calculated
        using the 'calc_floris_approx_table(...)' function described below, but
        can also be generated by hand using other tools like PyWake. df_approx
        typically has the form:

        df_approx=
                  wd    ws    ti    pow_000     pow_001  ...    pow_006 
        0        0.0   1.0    0.03      0.0         0.0  ...        0.0
        1        3.0   1.0    0.03      0.0         0.0  ...        0.0
        ...      ...   ...   ...        ...         ...  ...        ... 
        32399  357.0  24.0    0.18    5.0e6       5.0e6  ...       5.0e6
        32400  360.0  24.0    0.18    5.0e6       5.0e6  ...       5.0e6

        method (str, optional): Interpolation method, options are 'nearest' and
        'linear'. Defaults to 'linear'.
        wrap_0deg_to_360deg (bool, optional): The precalculated set of FLORIS solutions
        are typically calculates from 0 deg to 360 deg in steps of 2.0 deg or 3.0 deg.
        This means the last wind direction in the precalculated table of solutions is
        at 357.0 deg or 358.0 deg. If the user uses this for interpolation, any wind
        directions in 'df' with a value between 358.0 deg and 360.0 deg cannot be
        interpolated because it falls outside the bounds. This option copies the
        precalculated table solutions from 0 deg over to 360 deg to allow interpolation
        for the entire wind rose. Recommended to set to True. Defaults to True.
        extrapolate_ws (bool, optional): The precalculated set of FLORIS solutions,
        df_approx, only covers a finite range of wind speeds, typically from 1 m/s up
        to 25 m/s. Any wind speed values below or above this range therefore cannot 
        be interpolated using the 'linear' method and therefore becomes a NaN. To 
        prevent this, we can copy over the lowest and highest wind speed value interpolated
        to finite bounds to avoid this. For example, if our lowest wind speed calculated is
        1 m/s, we copy the solutions at 1 m/s over to a wind speed of 0 m/s, implicitly
        assuming these values are equal. This allows interpolation over wind speeds below
        1 m/s. Additionally, we copy the highest wind speed solutions (e.g., 25 m/s) over
        to a wind speed of 99 m/s to allow interpolation of values up to 99 m/s.
        Defaults to True.
        extrapolate_ti (bool, optional): The precalculated set of FLORIS solutions,
        df_approx, only covers a finite range of turbulence intensities, typically from
        0.03 to 0.18, being respectively 3% and 18%. In the same fashion at
        'extrapolate_ws', we copy the lowest and highest turbulence intensity solutions
        over to 0.00 and 1.00 turbulence intensity, to cover all possible conditions
        we may find and to avoid any NaN interpolation. This implicitly makes the
        assumption that the solutions at 0% TI are equal to your solutions at 3% TI,
        and that your solutions at 100% TI are equal to your solutions at 18% TI.
        This may or may not be a valid assumption for your scenario. Defaults to True.
        mirror_nans (bool, optional): The raw data for which the FLORIS predictions are
        made may contain NaNs for specific turbines, e.g., due to sensor issues or due
        to irregular turbine behavior. By setting mirror_nans=True, the NaNs for turbines
        from the raw data will be copied such that NaNs in the raw data will also mean
        NaNs in the FLORIS predictions. Recommended to set this to True to ensure the
        remainder of the energy ratio analysis is a fair and accurate comparison. Defaults
        to True.
        verbose (bool, optional): Print warnings and information to the console.
        Defaults to True.

    Returns:
        df (pd.DataFrame): The Pandas Dataframe containing the timeseries 'wd', 'ws'
        and 'ti', plus the power productions (and potentially local inflow conditions)
        of the turbines interpolated from the precalculated solutions table. For example,

        df=
                 time                    wd     ws      ti    pow_000     pow_001  ...    pow_006 
        0       2018-01-01 00:10:00   213.1   7.81    0.08  1251108.2    825108.2  ...   725108.9
        1       2018-01-01 00:20:00   215.6   7.65    0.08  1202808.0    858161.8  ...   692111.2
        ...                     ...   ...      ...     ...         ...        ...  ...        ...
        52103   2018-12-31 23:30:00    15.6   11.0    0.08  4235128.7   3825108.4  ...  2725108.3
        52104   2018-12-31 23:40:00    15.3   11.1    0.08  3860281.3   3987634.7  ...  2957021.7
    """

    # Format dataframe and get number of turbines
    # df = df.reset_index(drop=('time' in df.columns))
    nturbs = fsut.get_num_turbines(df_approx)

    # Check input
    if mirror_nans:
        if not ("pow_000" in df.columns) or not ("ws_000" in df.columns):
            raise UserWarning("The option mirror_nans=True requires the raw data's wind speed and power measurements to be included in the dataframe 'df'.")
    else:
        print("Warning: not mirroring NaNs from the raw data to the FLORIS predictions. This may skew your energy ratios.")

    # Check if all values in df fall within the precalculated solutions ranges
    for col in ["wd", "ws", "ti"]:
        # Check if all columns are defined
        if col not in df.columns:
            raise ValueError("Your SCADA dataframe is missing a column called '{:s}'.".format(col))
        if col not in df_approx.columns:
            raise ValueError("Your precalculated solutions dataframe is missing a column called '{:s}'.".format(col))

        # Check if approximate solutions cover the entire problem space
        if (
            (df[col].min() < (df_approx[col].min() - 1.0e-6)) |
            (df[col].max() > (df_approx[col].max() + 1.0e-6))
        ):
            print("Warning: the values in df[{:s}] exceed the range in the precalculated solutions df_fi_approx[{:s}].".format(col, col))
            print("   minimum/maximum value in df:        ({:.3f}, {:.3f})".format(df[col].min(), df[col].max()))
            print("   minimum/maximum value in df_approx: ({:.3f}, {:.3f})".format(df_approx[col].min(), df_approx[col].max()))


    # Define which variables we must map from df_approx to df
    varnames = ['pow']
    if 'ws_000' in df_approx.columns:
        varnames.append('ws')
    if 'wd_000' in df_approx.columns:
        varnames.append('wd')
    if 'ti_000' in df_approx.columns:
        varnames.append('ti')

    # Map individual data entries to full DataFrame
    if verbose:
        print("Mapping the precalculated solutions " +
              "from FLORIS to the dataframe...")
        print("  Creating a gridded interpolant with " +
              "interpolation method '%s'." % method)

    # Make a copy from wd=0.0 deg to wd=360.0 deg for wrapping
    if wrap_0deg_to_360deg and (not (df_approx["wd"] > 359.999999).any()):
        if not np.any(df_approx["wd"] < 0.01):
            raise UserWarning("wrap_0deg_to_360deg is set to True but no solutions at wd=0 deg in the precalculated solution table.")
        df_subset = df_approx[df_approx["wd"] == 0.0].copy()
        df_subset["wd"] = 360.0
        df_approx = pd.concat([df_approx, df_subset], axis=0).reset_index(drop=True)

    # Copy TI to lower and upper bound
    if extrapolate_ti:
        df_ti_lb = df_approx.loc[df_approx["ti"] == df_approx['ti'].min()].copy()
        df_ti_ub = df_approx.loc[df_approx["ti"] == df_approx['ti'].max()].copy()
        df_ti_lb["ti"] = 0.0
        df_ti_ub["ti"] = 1.0
        df_approx = pd.concat(
            [df_approx, df_ti_lb, df_ti_ub],
            axis=0
        ).reset_index(drop=True)

    # Copy WS to lower and upper bound
    if extrapolate_ws:
        df_ws_lb = df_approx.loc[df_approx["ws"] == df_approx['ws'].min()].copy()
        df_ws_ub = df_approx.loc[df_approx["ws"] == df_approx['ws'].max()].copy()
        df_ws_lb["ws"] = 0.0
        df_ws_ub["ws"] = 99.0
        df_approx = pd.concat(
            [df_approx, df_ws_lb, df_ws_ub],
            axis=0
        ).reset_index(drop=True)

    # Convert df_approx dataframe into a regular grid
    wd_array_approx = np.sort(df_approx["wd"].unique())
    ws_array_approx = np.sort(df_approx["ws"].unique())
    ti_array_approx = np.sort(df_approx["ti"].unique())
    xg, yg, zg = np.meshgrid(
        wd_array_approx,
        ws_array_approx,
        ti_array_approx,
        indexing='ij',
    )

    grid_dict = dict()
    for varname in varnames:
        colnames = ['{:s}_{:03d}'.format(varname, ti) for ti in range(nturbs)]
        f = interpolate.NearestNDInterpolator(
            df_approx[["wd", "ws", "ti"]],
            df_approx[colnames]
        )
        grid_dict["{:s}".format(varname)] = f(xg, yg, zg)

    # Prepare an minimal output dataframe
    cols_to_copy = ["wd", "ws", "ti"]
    if "time" in df.columns:
        cols_to_copy.append("time")
    df_out = df[cols_to_copy].copy()

    # Use interpolant to determine values for all turbines and variables
    for ii, varname in enumerate(varnames):
        if verbose:
            print('     Interpolating ' + varname + ' for all turbines...')
        colnames = ['{:s}_{:03d}'.format(varname, ti) for ti in range(nturbs)]

        if ii == 0:
            f = interpolate.RegularGridInterpolator(
                points=(wd_array_approx, ws_array_approx, ti_array_approx),
                values=grid_dict[varname],
                method=method,
                bounds_error=False,
            )
        else:
            f.values = np.array(grid_dict[varname], dtype=float)

        df_out.loc[df_out.index, colnames] = f(df[['wd', 'ws', 'ti']])

        if mirror_nans:
            # Copy NaNs in the raw data to the FLORIS predictions
            for c in colnames:
                if c in df.columns:
                    df_out.loc[df[c].isna(), c] = None
                else:
                    df_out.loc[:, c] = None

    return df_out


def calc_floris_approx_table(
    fi,
    wd_array=np.arange(0.0, 360.0, 1.0),
    ws_array=np.arange(1.0, 25.01, 1.0),
    ti_array=np.arange(0.03, 0.1801, 0.03),
    save_turbine_inflow_conditions_to_df=False,
    ):
    """This function calculates a large number of floris solutions for a rectangular grid
    of wind directions ('wd_array'), wind speeds ('ws_array'), and optionally turbulence
    intensities ('ti_array'). The variables that are saved are each turbine's power
    production, and optionally also each turbine's inflow wind direction, wind speed and
    turbulence intensity if 'save_turbine_inflow_conditions_to_df==True'.

    Args:
        fi (FlorisInterface): FlorisInterface object.
        wd_array (array, optional): Array of wind directions to evaluate in [deg]. This expands with the
          number of wind speeds and turbulence intensities. Defaults to np.arange(0.0, 360.0, 1.0).
        ws_array (array, optional): Array of wind speeds to evaluate in [m/s]. This expands with the
          number of wind directions and turbulence intensities. Defaults to np.arange(1.0, 25.01, 1.0).
        ti_array (array, optional): Array of turbulence intensities to evaluate in [-]. This expands with the
          number of wind directions and wind speeds. Defaults to np.arange(0.03, 0.1801, 0.03).
        save_turbine_inflow_conditions_to_df (bool, optional): When set to True, will also write each turbine's
        inflow wind direction, wind speed and turbulence intensity to the output dataframe. This increases the
        dataframe size but can provide useful information. Defaults to False.

    Returns:
        df_approx (pd.DataFrame): A Pandas DataFrame containing the floris simulation results for all wind
          direction, wind speed and turbulence intensity combinations. The outputs are the power production
          for each turbine, 'pow_000' until 'pow_{nturbs-1}', and optionally als each turbine's inflow wind
          direction, wind speed and turbulence intensity when save_turbine_inflow_conditions_to_df==True.

        Example for a 7-turbine floris object with
            wd_array=np.arange(0.0, 360.0, 3.0)
            ws_array=np.arange(1.0, 25.001, 1.0)
            ti_array=np.arange(0.03, 0.1801, 0.03)
            save_turbine_inflow_conditions_to_df=True

        Yields:
        
        df_approx=
                  wd    ws    ti    pow_000     ws_000  wd_000  ti_000  pow_001  ...    pow_006     ws_006  wd_006  ti_006
        0        0.0   1.0    0.03      0.0      1.0       0.0     0.03     0.0  ...        0.0      1.0       0.0     0.03
        1        3.0   1.0    0.03      0.0      1.0       3.0     0.03     0.0  ...        0.0      1.0       3.0     0.03
        2        6.0   1.0    0.03      0.0      1.0       6.0     0.03     0.0  ...        0.0      1.0       6.0     0.03
        3        9.0   1.0    0.03      0.0      1.0       9.0     0.03     0.0  ...        0.0      1.0       9.0     0.03
        4       12.0   1.0    0.03      0.0      1.0      12.0     0.03     0.0  ...        0.0      1.0      12.0     0.03
        ...      ...   ...   ...        ...        ...     ...     ...           ...        ...        ...     ...     ...
        32395  345.0  25.0    0.18      0.0  24.880843   345.0     0.18     0.0  ...        0.0  24.881165   345.0     0.18
        32396  348.0  25.0    0.18      0.0  24.880781   348.0     0.18     0.0  ...        0.0  24.881165   348.0     0.18
        32397  351.0  25.0    0.18      0.0  24.880755   351.0     0.18     0.0  ...        0.0  24.881165   351.0     0.18
        32398  354.0  25.0    0.18      0.0  24.880772   354.0     0.18     0.0  ...        0.0  24.881165   354.0     0.18
        32399  357.0  25.0    0.18      0.0  24.880829   357.0     0.18     0.0  ...        0.0  24.881165   357.0     0.18
        32400  360.0  25.0    0.18      0.0  24.880829   360.0     0.18     0.0  ...        0.0  24.881165   360.0     0.18
    """

    # if ti_array is None, use the current value in the FLORIS object
    if ti_array is None:
        ti = fi.floris.flow_field.turbulence_intensity
        ti_array = np.array([ti], dtype=float)

    fi = fi.copy()  # Create independent copy that we can manipulate
    num_turbines = len(fi.layout_x)

    # Format input arrays
    wd_array = np.sort(wd_array)
    ws_array = np.sort(ws_array)
    ti_array = np.sort(ti_array)
    wd_mesh, ws_mesh = np.meshgrid(wd_array, ws_array, indexing='ij')
    N_approx = len(wd_array) * len(ws_array) * len(ti_array)
    print(
        'Generating a df_approx table of FLORIS solutions ' +
        'covering a total of {:d} cases.'.format(N_approx)
    )

    # Create solutions, one set per turbulence intensity
    df_list = []
    for turb_intensity in ti_array:
        # Calculate solutions
        fi.reinitialize(
            wind_directions=wd_array,
            wind_speeds=ws_array,
            turbulence_intensity=turb_intensity,
        )
        fi.calculate_wake()
        turbine_powers = fi.get_turbine_powers()

        # Create a dictionary to save solutions in
        solutions_dict = {"wd": wd_mesh.flatten(), "ws": ws_mesh.flatten()}
        solutions_dict["ti"] = turb_intensity * np.ones(len(wd_array) * len(ws_array))
        for turbi in range(num_turbines):
            solutions_dict["pow_{:03d}".format(turbi)] = turbine_powers[:, :, turbi].flatten()
            if save_turbine_inflow_conditions_to_df:
                solutions_dict["ws_{:03d}".format(turbi)] = \
                    fi.floris.flow_field.u.mean(axis=4).mean(axis=3)[:, :, turbi].flatten()
                solutions_dict["wd_{:03d}".format(turbi)] = \
                    wd_mesh.flatten()  # Uniform wind direction
                solutions_dict["ti_{:03d}".format(turbi)] = \
                    fi.floris.flow_field.turbulence_intensity_field[:, :, turbi].flatten()
        df_list.append(pd.DataFrame(solutions_dict))

    print('Finished calculating the FLORIS solutions for the dataframe.')
    df_approx = pd.concat(df_list, axis=0).sort_values(by=["ti", "ws", "wd"])
    df_approx = df_approx.reset_index(drop=True)

    return df_approx


def add_gaussian_blending_to_floris_approx_table(df_fi_approx, wd_std=3.0, pdf_cutoff=0.995):
    """This function applies a Gaussian blending across the wind direction for the predicted
    turbine power productions from FLORIS. This is a post-processing step and achieves the
    same result as evaluating FLORIS directly with the UncertaintyInterface module. However,
    having this as a postprocess step allows for rapid generation of the FLORIS solutions for
    different values of wd_std without having to re-run FLORIS.

    Args:
        df_fi_approx (pd.DataFrame): Pandas DataFrame with precalculated FLORIS solutions,
          typically generated using flasc.floris_tools.calc_floris_approx_table().
        wd_std (float, optional): Standard deviation of the Gaussian blur that is applied
          across the wind direction in degrees. Defaults to 3.0.
        pdf_cutoff (float, optional): Cut-off point of the probability density function of
          the Gaussian curve. Defaults to 0.995 and thereby includes three standard
          deviations to the left and to the right of the evaluation.

    Returns:
        df_fi_approx_gauss (pd.DataFrame): Pandas DataFrame with Gaussian-blurred precalculated
          FLORIS solutions. The DataFrame typically has the columns "wd", "ws", "ti", and
          "pow_000" until "pow_{nturbs-1}", with nturbs being the number of turbines.

    """
    # Assume the resolution to be equal to the resolution of the wind direction steps
    wd_steps = np.unique(np.diff(np.unique(df_fi_approx["wd"])))
    pmf_res = wd_steps[0]

    # Set-up Gaussian kernel
    wd_bnd = int(np.ceil(norm.ppf(pdf_cutoff, scale=wd_std) / pmf_res))
    bound = wd_bnd * pmf_res
    wd_unc = np.linspace(-1 * bound, bound, 2 * wd_bnd + 1)
    wd_unc_pmf = norm.pdf(wd_unc, scale=wd_std)
    wd_unc_pmf /= np.sum(wd_unc_pmf)  # normalize so sum = 1.0

    # Map solutions to the right shape using a NN interpolant
    F = interpolate.NearestNDInterpolator(
        x=df_fi_approx[["wd", "ws", "ti"]],
        y=df_fi_approx[[c for c in df_fi_approx.columns if "pow_" in c]]
    )
    
    # Create new sets to interpolate over for Gaussian kernel
    wd = df_fi_approx["wd"]
    wd = wrap_360(np.tile(wd, (len(wd_unc), 1)).T + np.tile(wd_unc, (wd.shape[0], 1)))

    ws = df_fi_approx["ws"]
    ws = np.tile(ws, (len(wd_unc), 1)).T

    ti = df_fi_approx["ti"]
    ti = np.tile(ti, (len(wd_unc), 1)).T

    # Interpolate power values
    turbine_powers = F(wd, ws, ti)
    weights = np.tile(wd_unc_pmf[None, :, None], (turbine_powers.shape[0], 1, turbine_powers.shape[2]))
    turbine_powers_gaussian = np.sum(weights * turbine_powers, axis=1)  # Weighted sum

    pow_cols = [c for c in df_fi_approx.columns if c.startswith("pow_")]
    df_fi_approx_gauss = pd.concat(
        [
            df_fi_approx[["wd", "ws", "ti"]],
            pd.DataFrame(dict(zip(pow_cols, turbine_powers_gaussian.T)))
        ],
        axis=1
    )
    
    return df_fi_approx_gauss


def get_turbs_in_radius(x_turbs, y_turbs, turb_no, max_radius,
                        include_itself, sort_by_distance=False):
    """Determine which turbines are within a certain radius of other
    wind turbines.

    Args:
        x_turbs ([list, array]): Long. locations of turbines
        y_turbs ([list, array]): Lat. locations of turbines
        turb_no ([int]): Turbine number for which the distance is
        calculated w.r.t. the other turbines.
        max_radius ([float]): Maximum distance between turbines to be
        considered within the radius of turbine [turb_no].
        include_itself ([type]): Include itself in the list of turbines
        within the radius.
        sort_by_distance (bool, optional): Sort the output list of turbines
        according to distance to the turbine, from closest to furthest (but
        still within radius). Defaults to False.

    Returns:
        turbs_within_radius ([list]): List of turbines that are within the
        prespecified distance from turbine [turb_no].
    """
    dr_turb = np.sqrt((x_turbs - x_turbs[turb_no])**2.0 +
                      (y_turbs - y_turbs[turb_no])**2.0)
    turbine_list = np.array(range(len(x_turbs)))

    if sort_by_distance:
        indices_sorted = np.argsort(dr_turb)
        dr_turb = dr_turb[indices_sorted]
        turbine_list = turbine_list[indices_sorted]

    turbs_within_radius = turbine_list[dr_turb <= max_radius]
    if not include_itself:
        turbs_within_radius = [ti for ti in turbs_within_radius
                               if not ti == turb_no]

    return turbs_within_radius


def get_upstream_turbs_floris(fi, wd_step=0.1, wake_slope=0.10,
                              plot_lines=False):
    """Determine which turbines are operating in freestream (unwaked)
    flow, for the entire wind rose. This function will return a data-
    frame where each row will present a wind direction range and a set
    of wind turbine numbers for which those turbines are operating
    upstream. This is useful in determining the freestream conditions.

    Args:
        fi ([floris object]): FLORIS object of the farm of interest.
        wd_step (float, optional): Wind direction discretization step.
        It will test what the upstream turbines are every [wd_step]
        degrees. A lower number means more accurate results, but
        typically there's no real benefit below 2.0 deg or so.
        Defaults to 0.1.
        wake_slope (float, optional): linear slope of the wake (dy/dx)
        plot_lines (bool, optional): Enable plotting wakes/turbines.
        Defaults to False.

    Returns:
        df_upstream ([pd.DataFrame]): A Pandas Dataframe in which each row
        contains a wind direction range and a list of turbine numbers. For
        that particular wind direction range, the turbines numbered are
        all upstream according to the FLORIS predictions. Depending on
        the FLORIS model parameters and ambient conditions, these results
        may vary slightly. Though, having minimal wake losses should not
        noticably affect your outcomes. Empirically, this approach has
        yielded good results with real SCADA data for determining what
        turbines are waked/unwaked and has served useful for determining
        what turbines to use as reference.
    """

    # Get farm layout
    x = fi.layout_x
    y = fi.layout_y
    n_turbs = len(x)
    D = [t["rotor_diameter"] for t in fi.floris.farm.turbine_definitions]
    D = np.array(D, dtype=float)

    # Setup output list
    upstream_turbs_ids = []  # turbine numbers that are freestream
    upstream_turbs_wds = []  # lower bound of bin

    # Rotate farm and determine freestream/waked turbines
    for wd in np.arange(0., 360., wd_step):
        is_freestream = [True for _ in range(n_turbs)]
        x_rot = (np.cos((wd-270.) * np.pi / 180.) * x -
                 np.sin((wd-270.) * np.pi / 180.) * y)
        y_rot = (np.sin((wd-270.) * np.pi / 180.) * x +
                 np.cos((wd-270.) * np.pi / 180.) * y)

        if plot_lines:
            fig, ax = plt.subplots()
            for ii in range(n_turbs):
                ax.plot(x_rot[ii] * np.ones(2), [y_rot[ii] - D[ii] / 2, y_rot[ii] + D[ii] / 2], 'k')
            for ii in range(n_turbs):
                ax.text(x_rot[ii], y_rot[ii], 'T%03d' % ii)
            ax.axis('equal')

        srt = np.argsort(x_rot)
        x_rot_srt = x_rot[srt]
        y_rot_srt = y_rot[srt]
        for ii in range(n_turbs):
            x0 = x_rot_srt[ii]
            y0 = y_rot_srt[ii]

            def yw_upper(x):
                y = (y0 + D[ii]) + (x-x0) * wake_slope
                if isinstance(y, (float, np.float64, np.float32)):
                    if x < (x0 + 0.01):
                        y = -np.Inf
                else:
                    y[x < x0 + 0.01] = -np.Inf
                return y

            def yw_lower(x):
                y = (y0 - D[ii]) - (x-x0) * wake_slope
                if isinstance(y, (float, np.float64, np.float32)):
                    if x < (x0 + 0.01):
                        y = -np.Inf
                else:
                    y[x < x0 + 0.01] = -np.Inf
                return y

            is_in_wake = lambda xt, yt: ((yt < yw_upper(xt)) &
                                         (yt > yw_lower(xt)))

            is_freestream = (is_freestream &
                             ~is_in_wake(x_rot_srt, y_rot_srt + D/2.) &
                             ~is_in_wake(x_rot_srt, y_rot_srt - D/2.))

            if plot_lines:
                x1 = np.max(x_rot_srt) + 500.
                ax.fill_between([x0, x1, x1, x0],
                                [yw_upper(x0+0.02), yw_upper(x1),
                                 yw_lower(x1), yw_lower(x0+0.02)],
                                alpha=0.1, color='k', edgecolor=None)

        usrt = np.argsort(srt)
        is_freestream = is_freestream[usrt]
        turbs_freestream = list(np.where(is_freestream)[0])

        if len(upstream_turbs_wds) == 0:
            upstream_turbs_ids.append(turbs_freestream)
            upstream_turbs_wds.append(wd)
        elif not(turbs_freestream == upstream_turbs_ids[-1]):
            upstream_turbs_ids.append(turbs_freestream)
            upstream_turbs_wds.append(wd)

        if plot_lines:
            ax.set_title('wd = %03d' % wd)
            ax.set_xlim([np.min(x_rot)-500., x1])
            ax.set_ylim([np.min(y_rot)-500., np.max(y_rot)+500.])
            ax.plot(x_rot[turbs_freestream],
                    y_rot[turbs_freestream],
                    'o', color='green')

    # # Connect at 360 degrees
    # if upstream_turbs_ids[0] == upstream_turbs_ids[-1]:
    #     upstream_turbs_wds.pop(0)
    #     upstream_turbs_ids.pop(0)

    # Go from list to bins for upstream_turbs_wds
    upstream_turbs_wds = [[upstream_turbs_wds[i], upstream_turbs_wds[i+1]]
                          for i in range(len(upstream_turbs_wds)-1)]
    upstream_turbs_wds.append([upstream_turbs_wds[-1][-1], 360.])

    df_upstream = pd.DataFrame({'wd_min': [wd[0] for wd in upstream_turbs_wds],
                                'wd_max': [wd[1] for wd in upstream_turbs_wds],
                                'turbines': upstream_turbs_ids})

    return df_upstream

def get_dependent_turbines_by_wd(fi_in, test_turbine, 
    wd_array=np.arange(0., 360., 2.), change_threshold=0.001, limit_number=None, 
    ws_test=9., return_influence_magnitudes=False):
    """
    Computes all turbines that depend on the operation of a specified 
    turbine (test_turbine) for each wind direction in wd_array, using 
    the FLORIS model specified by fi_in to detect dependencies. 

    Args:
        fi ([floris object]): FLORIS object of the farm of interest.
        test_turbine ([int]): Turbine for which dependencies are found.
        wd_array ([np.array]): Wind directions at which to determine 
            dependencies. Defaults to [0, 2, ... , 358].
        change_threshold (float): Fractional change in power needed 
            to denote a dependency. Defaults to 0. (any change in power 
            is marked as a dependency)
        limit_number (int | NoneType): Number of turbines that a 
            turbine can have as dependencies. If None, returns all 
            turbines that depend on each turbine. Defaults to None.
        ws_test (float): Wind speed at which FLORIS model is run to 
            determine dependencies.  Defaults to 9. m/s.
        return_influence_magnitudes (Bool): Flag for whether to return 
            an array containing the magnitude of the influence of the 
            test_turbine on all turbines.
        
    Returns:
        dep_indices_by_wd (list): A 2-dimensional list. Each element of 
            the outer level list, which represents wind direction, 
            contains a list of the turbines that depend on test_turbine 
            for that wind direction. The second-level list may be empty 
            if no turbine depends on the test_turbine for that wind 
            direciton (e.g., the turbine is in the back row).
        all_influence_magnitudes ([np.array]): 2-D numpy array of 
            influences of test_turbine on all other turbines, with size 
            (number of wind directions) x (number of turbines). Returned
            only if return_influence_magnitudes is True.
    """
    # Copy fi to a local to not mess with incoming
    fi = copy.deepcopy(fi_in)
    
    # Compute the base power
    fi.reinitialize(
        wind_speeds=[ws_test], 
        wind_directions=wd_array
    )
    fi.calculate_wake()
    base_power = fi.get_turbine_powers()[:,0,:] # remove unneeded dimension
    
    # Compute the test power
    if len(fi.floris.farm.turbine_type) > 1:
        # Remove test turbine from list
        fi.floris.farm.turbine_type.pop(test_turbine) 
    else: # Only a single turbine type defined for the whole farm; do nothing
        pass
    fi.reinitialize(
        layout_x=np.delete(fi.layout_x, [test_turbine]),
        layout_y=np.delete(fi.layout_y, [test_turbine]),
        wind_speeds=[ws_test],
        wind_directions=wd_array
    ) # This will reindex the turbines; undone in following steps.
    fi.calculate_wake()
    test_power = fi.get_turbine_powers()[:,0,:] # remove unneeded dimension
    test_power = np.insert(test_power, test_turbine, 
        base_power[:,test_turbine], axis=1)

    if return_influence_magnitudes:
        all_influence_magnitudes = np.zeros_like(test_power)
    
    # Find the indices that have changed
    dep_indices_by_wd = [None]*len(wd_array)
    for i in range(len(wd_array)):
        all_influences = np.abs(test_power[i,:] - base_power[i,:])/\
                         base_power[i,:]
        # Sort with highest influence first; trim to limit_number
        influence_order = np.flip(np.argsort(all_influences))[:limit_number]
        # Mask to only those that meet the threshold
        influence_order = influence_order[
            all_influences[influence_order] >= change_threshold
        ]
        
        # Store in output
        dep_indices_by_wd[i] = list(influence_order)
        if return_influence_magnitudes:
            all_influence_magnitudes[i,:] = all_influences
    

    # Remove the turbines own indice
    if return_influence_magnitudes:
        return dep_indices_by_wd, all_influence_magnitudes
    else:
        return dep_indices_by_wd

def get_all_dependent_turbines(fi_in, wd_array=np.arange(0., 360., 2.), 
    change_threshold=0.001, limit_number=None, ws_test=9.):
    """
    Wrapper for get_dependent_turbines_by_wd() that loops over all 
    turbines in the farm and packages their dependencies as a pandas 
    dataframe.

    Args:
        fi ([floris object]): FLORIS object of the farm of interest.
        wd_array ([np.array]): Wind directions at which to determine 
            dependencies. Defaults to [0, 2, ... , 358].
        change_threshold (float): Fractional change in power needed 
            to denote a dependency. Defaults to 0. (any change in power 
            is marked as a dependency)
        limit_number (int | NoneType): Number of turbines that a 
            turbine can have as dependencies. If None, returns all 
            turbines that depend on each turbine. Defaults to None.
        ws_test (float): Wind speed at which FLORIS model is run to 
            determine dependencies. Defaults to 9. m/s.
        
    Returns:
        df_out ([pd.DataFrame]): A Pandas Dataframe in which each row
            contains a wind direction, each column is a turbine, and 
            each entry is the turbines that depend on the column turbine 
            at the row wind direction. Dependencies can be extracted 
            as: For wind direction wd, the turbines that depend on 
            turbine T are df_out.loc[wd, T]. Dependencies are ordered, 
            with strongest dependencies appearing first.
    """

    results = []
    for t_i in range(len(fi_in.layout_x)):
        results.append(
            get_dependent_turbines_by_wd(
                fi_in, t_i, wd_array, change_threshold, limit_number, ws_test
            )
        )
    
    df_out = (pd.DataFrame(data=results, columns=wd_array)
              .transpose()
              .reset_index().rename(columns={"index":"wd"}).set_index("wd")
             )
    
    return df_out

def get_all_impacting_turbines(fi_in, wd_array=np.arange(0., 360., 2.), 
    change_threshold=0.001, limit_number=None, ws_test=9.):
    """
    Calculate which turbines impact a specified turbine based on the 
    FLORIS model. Essentially a wrapper for 
    get_dependent_turbines_by_wd() that loops over all turbines and 
    extracts their impact magnitudes, then sorts.

    Args:
        fi ([floris object]): FLORIS object of the farm of interest.
        wd_array ([np.array]): Wind directions at which to determine 
            dependencies. Defaults to [0, 2, ... , 358].
        change_threshold (float): Fractional change in power needed 
            to denote a dependency. Defaults to 0. (any change in power 
            is marked as a dependency)
        limit_number (int | NoneType): Number of turbines that a 
            turbine can depend on. If None, returns all 
            turbines that each turbine depends on. Defaults to None.
        ws_test (float): Wind speed at which FLORIS model is run to 
            determine dependencies. Defaults to 9. m/s.

    Returns:
        df_out ([pd.DataFrame]): A Pandas Dataframe in which each row
            contains a wind direction, each column is a turbine, and 
            each entry is the turbines that the column turbine depends 
            on at the row wind direction. Dependencies can be extracted 
            as: For wind direction wd, the turbines that impact turbine 
            T are df_out.loc[wd, T]. Impacting turbines are simply 
            ordered by magnitude of impact.
    """

    dependency_magnitudes = np.zeros(
        (len(wd_array),len(fi_in.layout_x),len(fi_in.layout_x))
    )
    
    for t_i in range(len(fi_in.layout_x)):
        _, ti_dep_mags = get_dependent_turbines_by_wd(
                fi_in, t_i, wd_array, change_threshold, limit_number, ws_test,
                return_influence_magnitudes=True
            )
        dependency_magnitudes[:,:,t_i] = ti_dep_mags
    
    # Sort
    impact_order = np.flip(np.argsort(dependency_magnitudes, axis=2), axis=2)

    # Truncate to limit_number
    impact_order = impact_order[:,:,:limit_number]

    # Build up multi-level results list
    results = []

    for wd in range(len(wd_array)):
        wd_results = []
        for t_j in range(len(fi_in.layout_x)):
            impacts_on_t_j = dependency_magnitudes[wd, t_j, :]
            impact_order_t_j = impact_order[wd, t_j, :]
            impact_order_t_j = impact_order_t_j[
                impacts_on_t_j[impact_order_t_j] >= change_threshold
            ]
            wd_results.append(list(impact_order_t_j))
        results.append(wd_results)

    # Convert to dataframe
    df_out = (pd.DataFrame(data=results, index=wd_array)
            .reset_index().rename(columns={"index":"wd"}).set_index("wd")
            )

    return df_out

# Wrapper function to easily set new TI values
def _fi_set_ws_wd_ti(fi, wd=None, ws=None, ti=None):
    nturbs = len(fi.layout_x)

    # Convert scalar values to lists
    if not isinstance(wd, list):
        if isinstance(wd, np.ndarray):
            wd = list(wd)
        elif wd is not None:
            wd = list(np.repeat(wd, nturbs))
    if not isinstance(ws, list):
        if isinstance(ws, np.ndarray):
            ws = list(ws)
        elif ws is not None:
            ws = list(np.repeat(ws, nturbs))
    if not isinstance(ti, list):
        if isinstance(ti, np.ndarray):
            ti = list(ti)
        elif ti is not None:
            ti = list(np.repeat(ti, nturbs))

    wind_layout = (np.array(fi.layout_x), np.array(fi.layout_y))

    fi.reinitialize_flow_field(
        wind_layout=wind_layout,
        wind_direction=wd,
        wind_speed=ws,
        turbulence_intensity=ti
    )
    return fi
