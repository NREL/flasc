"""Module for tuning FLORIS to SCADA data."""

# This is a preliminary implementation of tuning methods for FLORIS to SCADA.
# The code is focused on methods for the Empirical Guassian wake model and is
# based on contributions from Elizabeth Eyeson, Paul Fleming (paul.fleming@nrel.gov)
# Misha Sinner (michael.sinner@nrel.gov) and Eric Simley at NREL, and Bart
# Doekemeijer at Shell, as well as discussions with Diederik van Binsbergen at
# NTNU. Please see readme.txt for more information.


import numpy as np
import polars as pl

import flasc.utilities.floris_tools as ftools
from flasc.analysis import energy_ratio as er, total_uplift as tup
from flasc.analysis.energy_ratio_input import EnergyRatioInput
from flasc.utilities.energy_ratio_utilities import add_power_ref, add_power_test
from flasc.utilities.tuner_utilities import replicate_nan_values, resim_floris


def evaluate_overall_wake_loss(df_, df_freq=None):
    """Evaluate the overall wake loss from pow_ref to pow_test as percent reductions.

    Args:
        df_ (pl.DataFrame): Polars dataframe possibly containing Null values
        df_freq (Dataframe): Not yet used

    Returns:
        float: Overall wake losses

    """
    # Not sure yet if we want to figure out how to use df_freq here
    return 100 * (df_["pow_ref"].sum() - df_["pow_test"].sum()) / df_["pow_ref"].sum()


def sweep_velocity_model_parameter_for_overall_wake_losses(
    parameter,
    value_candidates,
    df_scada_in,
    fm_in,
    ref_turbines,
    test_turbines,
    param_idx=None,
    yaw_angles=None,
    wd_min=0.0,
    wd_max=360.0,
    # ws_step: float = 1.0,
    ws_min=0.0,
    ws_max=50.0,
    df_freq=None,  # Not yet certain we will use this
):
    """Sweep the parameter in FLORIS using the values in value_candidates.

    Compare to SCADA data in df_scada_in using the overall_wake_loss

    Args:
        parameter (str): The parameter to sweep
        value_candidates (list): The values to sweep
        df_scada_in (DataFrame): The SCADA data
        fm_in (floris.tools.Floris): The FLORIS model
        ref_turbines (list): The reference turbines
        test_turbines (list): The test turbines
        param_idx (int): The parameter index
        yaw_angles (np.ndarray): The yaw angles
        wd_min (float): The minimum wind direction
        wd_max (float): The maximum wind direction
        ws_min (float): The minimum wind speed
        ws_max (float): The maximum wind speed
        df_freq (DataFrame): The frequency data

    Returns:
        A tuple (np.ndarray, np.ndarray) where the first element is the FLORIS wake losses
        and the second element is the SCADA wake losses
    """
    # Currently assuming pow_ref and pow_test already assigned
    # Also assuming limit to ws/wd range accomplished but could revisit?

    # Assign the ref and test cols
    df_scada = pl.from_pandas(df_scada_in)

    df_scada = df_scada.with_columns(
        pl.Series(name="simple_index", values=np.arange(0, df_scada.shape[0]))
    )

    # Trim to ws/wd
    df_scada = df_scada.filter(
        (pl.col("ws") >= ws_min)  # Filter the mean wind speed
        & (pl.col("ws") < ws_max)
        & (pl.col("wd") >= wd_min)  # Filter the mean wind direction
        & (pl.col("wd") < wd_max)
    )

    # Trim the yaw angle matrices
    if yaw_angles is not None:
        simple_index = df_scada["simple_index"].to_numpy()
        yaw_angles = yaw_angles[simple_index, :, :]

    ref_cols = [f"pow_{i:03d}" for i in ref_turbines]
    test_cols = [f"pow_{i:03d}" for i in test_turbines]
    df_scada = add_power_ref(df_scada, ref_cols)
    df_scada = add_power_test(df_scada, test_cols)

    # First collect the scada wake loss
    scada_wake_loss = evaluate_overall_wake_loss(df_scada)

    # Now loop over FLORIS candidates and collect the wake loss
    floris_wake_losses = np.zeros(len(value_candidates))
    fm = fm_in.copy()
    for idx, vc in enumerate(value_candidates):
        # Set the parameter
        fm.set_param(parameter, vc, param_idx)

        # Collect the FLORIS results
        df_floris = resim_floris(fm, df_scada.to_pandas(), yaw_angles=yaw_angles)
        df_floris = pl.from_pandas(df_floris)

        # Assign the ref and test cols
        df_floris = add_power_ref(df_floris, ref_cols)
        df_floris = add_power_test(df_floris, test_cols)

        # Get the wake loss
        floris_wake_losses[idx] = evaluate_overall_wake_loss(df_floris)

    # Return the error
    return floris_wake_losses, scada_wake_loss


def select_best_wake_model_parameter(floris_results, scada_results, value_candidates, ax=None):
    """Determine the best fit with respect to squared error.

    Args:
        floris_results (np.ndarray): The FLORIS wake losses
        scada_results (np.ndarray): The SCADA wake losses
        value_candidates (np.ndarray): The parameter values
        ax (Axes): The axes to plot on.  If None, no plot is made.
            Default is None.

    Returns:
        float: best fit parameter value
    """
    error_values = (floris_results - scada_results) ** 2

    best_param = value_candidates[np.argmin(error_values)]
    best_floris_result = floris_results[np.argmin(error_values)]

    if ax is not None:
        ax.plot(value_candidates, floris_results, "b.-", label="FLORIS")
        ax.scatter(best_param, best_floris_result, color="r", marker="o", label="Best Fit")
        ax.axhline(scada_results, color="k", label="SCADA")
        ax.grid(True)
        ax.legend()

    return best_param


def sweep_wd_std_for_er(
    value_candidates,
    df_scada_in,
    df_approx_,
    ref_turbines,
    test_turbines,
    yaw_angles=None,
    wd_step=2.0,
    wd_min=0.0,
    wd_max=360.0,
    ws_step: float = 1.0,
    ws_min=0.0,
    ws_max=50.0,
    bin_cols_in=["wd_bin", "ws_bin"],
    weight_by="min",  # min, sum
    df_freq=None,  # Not yet certain we will use this,
    remove_all_nulls=False,
):
    """Determine the best-fit wd_std for FLORIS by comparison with energy ratio plots.

    TODO: Reimplement that comparison only takes place when FLORIS value is below some threshold

    Args:
        value_candidates (list): The values to sweep
        df_scada_in (DataFrame): The SCADA data
        df_approx_ (DataFrame): The FLORIS approximation data
        ref_turbines (list): The reference turbines
        test_turbines (list): The test turbines
        yaw_angles (np.ndarray): The yaw angles
        wd_step (float): The wind direction step
        wd_min (float): The minimum wind direction
        wd_max (float): The maximum wind direction
        ws_step (float): The wind speed step
        ws_min (float): The minimum wind speed
        ws_max (float): The maximum wind speed
        bin_cols_in (list): The bin columns
        weight_by (str): The weight method.  Can be 'min' or 'sum'.
            Default is 'min'.
        df_freq (DataFrame): The frequency data
        remove_all_nulls (bool): Remove all nulls.  Default is False.

    Returns:
        A tuple (np.ndarray, list): The first element is the FLORIS energy ratio errors
            and the second element is the dataframes.

    """
    # Currently assuming pow_ref and pow_test already assigned
    # Also assuming limit to ws/wd range accomplished but could revisit?

    # Assign the ref and test cols
    df_scada = pl.from_pandas(df_scada_in)

    # Trim to ws/wd
    df_scada = df_scada.filter(
        (pl.col("ws") >= ws_min)  # Filter the mean wind speed
        & (pl.col("ws") < ws_max)
        & (pl.col("wd") >= wd_min)  # Filter the mean wind direction
        & (pl.col("wd") < wd_max)
    )

    ref_cols = [f"pow_{i:03d}" for i in ref_turbines]
    test_cols = [f"pow_{i:03d}" for i in test_turbines]
    df_scada = add_power_ref(df_scada, ref_cols)
    df_scada = add_power_test(df_scada, test_cols)

    df_scada = df_scada.to_pandas()
    df_scada["ti"] = 0.1

    # Now loop over FLORIS candidates and collect the wake loss
    er_error = np.zeros(len(value_candidates))
    df_list = []
    for idx, wd_std in enumerate(value_candidates):
        if wd_std > 0:
            df_approx_wd_std = ftools.add_gaussian_blending_to_floris_approx_table(
                df_approx_, wd_std
            )
        else:
            df_approx_wd_std = df_approx_.copy()

        df_floris = ftools.interpolate_floris_from_df_approx(
            df_scada, df_approx_wd_std, mirror_nans=False, wrap_0deg_to_360deg=False
        )
        df_floris = replicate_nan_values(df_scada, df_floris)

        # Collect the FLORIS results
        df_floris = pl.from_pandas(df_floris)

        # Assign the ref and test cols
        df_floris = add_power_ref(df_floris, ref_cols)
        df_floris = add_power_test(df_floris, test_cols)

        # Compare the energy ratio to SCADA
        er_in = EnergyRatioInput([df_scada, df_floris.to_pandas()], ["SCADA", "FLORIS"])

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

        df_ = er_out.df_result.copy()

        df_list.append(df_)

        # Grab the energy ratios and counts
        scada_vals = df_["SCADA"].values
        floris_vals = df_["FLORIS"].values
        count_vals = df_["count_SCADA"].values

        er_error[idx] = ((scada_vals - floris_vals) ** 2 * count_vals).sum() / count_vals.sum()

    # Return the error
    return er_error, df_list


def select_best_wd_std(er_results, value_candidates, ax=None):
    """Consider wd_std and determine the best fit with respect to squared error.

    Args:
        er_results (np.ndarray): The energy ratio errors
        value_candidates (np.ndarray): The parameter values
        ax (Axes): The axes to plot on.  If None, no plot is made.
            Default is None.

    Returns:
        float: The best parameter value
    """
    error_sq = er_results**2

    best_param = value_candidates[np.argmin(error_sq)]

    if ax is not None:
        ax.plot(value_candidates, error_sq, "b.-", label="Energy Ratio Error")
        ax.axvline(best_param, color="r")
        ax.set_xlabel("wd_std")
        ax.set_ylabel("squared error")
        ax.grid(True)
        ax.legend()

    return best_param


def sweep_deflection_parameter_for_total_uplift(
    parameter,
    value_candidates,
    df_scada_baseline_in,
    df_scada_wakesteering_in,
    fm_in,
    ref_turbines,
    test_turbines,
    yaw_angles_baseline=None,
    yaw_angles_wakesteering=None,
    wd_step=2.0,
    wd_min=0.0,
    wd_max=360.0,
    ws_step: float = 1.0,
    ws_min=0.0,
    ws_max=50.0,
    bin_cols_in=["wd_bin", "ws_bin"],
    weight_by="min",  # min, sum
    df_freq=None,  # Not yet certain we will use this,
    remove_all_nulls=False,
):
    """Sweep values of the deflection parameter in FLORIS and compare to SCADA.

    Comparison is made wrt overall uplift.

    Args:
        parameter (str): The parameter to sweep
        value_candidates (list): The values to sweep
        df_scada_baseline_in (DataFrame): The baseline SCADA data
        df_scada_wakesteering_in (DataFrame): The wake steering SCADA data
        fm_in (floris.tools.Floris): The FLORIS model
        ref_turbines (list): The reference turbines
        test_turbines (list): The test turbines
        yaw_angles_baseline (np.ndarray): The yaw angles for the baseline
        yaw_angles_wakesteering (np.ndarray): The yaw angles for the wake steering
        wd_step (float): The wind direction step
        wd_min (float): The minimum wind direction
        wd_max (float): The maximum wind direction
        ws_step (float): The wind speed step
        ws_min (float): The minimum wind speed
        ws_max (float): The maximum wind speed
        bin_cols_in (list): The bin columns
        weight_by (str): The weight method.  Can be 'min' or 'sum'.
            Default is 'min'.
        df_freq (DataFrame): The frequency data
        remove_all_nulls (bool): Remove all nulls.  Default is False.

    Returns:
        A typle (np.ndarray, np.ndarray) where the first element is the FLORIS total uplifts
        and the second element is the SCADA total uplifts
    """
    # Currently assuming pow_ref and pow_test already assigned
    # Also assuming limit to ws/wd range accomplished but could revisit?

    # Assign the ref and test cols
    df_scada_baseline = pl.from_pandas(df_scada_baseline_in)
    df_scada_wakesteering = pl.from_pandas(df_scada_wakesteering_in)

    # Add a simple index
    df_scada_baseline = df_scada_baseline.with_columns(
        pl.Series(name="simple_index", values=np.arange(0, df_scada_baseline.shape[0]))
    )
    df_scada_wakesteering = df_scada_wakesteering.with_columns(
        pl.Series(name="simple_index", values=np.arange(0, df_scada_wakesteering.shape[0]))
    )

    # Trim to ws/wd
    df_scada_baseline = df_scada_baseline.filter(
        (pl.col("ws") >= ws_min)  # Filter the mean wind speed
        & (pl.col("ws") < ws_max)
        & (pl.col("wd") >= wd_min)  # Filter the mean wind direction
        & (pl.col("wd") < wd_max)
    )
    df_scada_wakesteering = df_scada_wakesteering.filter(
        (pl.col("ws") >= ws_min)  # Filter the mean wind speed
        & (pl.col("ws") < ws_max)
        & (pl.col("wd") >= wd_min)  # Filter the mean wind direction
        & (pl.col("wd") < wd_max)
    )

    # Trim the yaw angle matrices
    if yaw_angles_baseline is not None:
        simple_index = df_scada_baseline["simple_index"].to_numpy()
        yaw_angles_baseline = yaw_angles_baseline[simple_index, :]
    if yaw_angles_wakesteering is not None:
        simple_index = df_scada_wakesteering["simple_index"].to_numpy()
        yaw_angles_wakesteering = yaw_angles_wakesteering[simple_index, :]

    ref_cols = [f"pow_{i:03d}" for i in ref_turbines]
    test_cols = [f"pow_{i:03d}" for i in test_turbines]
    df_scada_baseline = add_power_ref(df_scada_baseline, ref_cols)
    df_scada_baseline = add_power_test(df_scada_baseline, test_cols)
    df_scada_wakesteering = add_power_ref(df_scada_wakesteering, ref_cols)
    df_scada_wakesteering = add_power_test(df_scada_wakesteering, test_cols)

    df_scada_baseline = df_scada_baseline.to_pandas()
    df_scada_wakesteering = df_scada_wakesteering.to_pandas()

    # Compare the scada uplift
    er_in = EnergyRatioInput(
        [df_scada_baseline, df_scada_wakesteering], ["Baseline [SCADA]", "Controlled [SCADA]"]
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

    scada_uplift = scada_uplift_result["Uplift [SCADA]"]["energy_uplift_ctr_pc"]

    # Now loop over FLORIS candidates and collect the uplift
    floris_uplifts = np.zeros(len(value_candidates))
    # df_list = []
    for idx, vc in enumerate(value_candidates):
        # Set the parameter for baseline and wake steering
        fm_baseline = fm_in.copy()
        fm_baseline.set_param(parameter, vc)
        fm_wakesteering = fm_baseline.copy()

        # Collect the FLORIS results
        df_floris_baseline = resim_floris(
            fm_baseline, df_scada_baseline, yaw_angles=yaw_angles_baseline
        )
        df_floris_wakesteering = resim_floris(
            fm_wakesteering, df_scada_wakesteering, yaw_angles=yaw_angles_wakesteering
        )

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
            ["Baseline [FLORIS]", "Controlled [FLORIS]"],
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
