"""Module for cross-checking the consistency of the northing offset between turbines."""

from datetime import timedelta as td

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floris.utilities import wrap_360

from flasc.logging_manager import LoggingManager
from flasc.utilities import floris_tools as ftools, optimization as opt

logger_manager = LoggingManager()  # Instantiate LoggingManager
logger = logger_manager.logger  # Obtain the reusable logger


# TODO: This one is used but is it all fully up to date?
def crosscheck_northing_offset_consistency(
    df, fm, bias_timestep=td(days=120), nan_thrshld=0.50, plot_figure=True
):
    """Cross-check the consistency of the northing offset between turbines.

    Args:
        df (pd.DataFrame | FlascDataFrame): DataFrame containing the relevant data.
        fm (floris.simulation.Floris): Floris object.
        bias_timestep (timedelta, optional): Time step for bias calculation.
            Defaults to td(days=120).
        nan_thrshld (float, optional): Threshold for NaN values. Defaults to 0.50.
        plot_figure (bool, optional): Whether to plot the figure. Defaults to True.

    Returns:
        list: List of strings indicating the status of each turbine
    """
    # Load data and extract info
    num_turbines = len(fm.layout_x)
    turbine_list = range(num_turbines)

    # Set up time_array and split into chunks
    time_array = np.array(df["time"])
    t = pd.to_datetime(time_array[0])
    idx_chunks = list()
    while t < pd.to_datetime(time_array[-1]):
        t_next = t + bias_timestep
        idx_chunks.append((time_array >= t) & (time_array < t_next))
        t = t_next
    N_chnks = len(idx_chunks)

    # Get reference turbines and create placeholder dataframes
    N_rt = 5  # Number of reference turbines
    turbs_ref_list = [[] for _ in turbine_list]
    bias_output_list = [[] for _ in turbine_list]
    for ti in turbine_list:
        turbs_in_radius = ftools.get_turbs_in_radius(
            fm.layout_x,
            fm.layout_y,
            ti,
            max_radius=1.0e9,
            include_itself=False,
            sort_by_distance=True,
        )
        turbs_ref_list[ti] = turbs_in_radius[0:N_rt]
        bias_output_list[ti] = pd.DataFrame(
            data=np.reshape(np.full(N_chnks * N_rt, np.nan), (N_chnks, N_rt)),
            columns=["T%03d" % ti_ref for ti_ref in turbs_ref_list[ti]],
        )

    for ti in turbine_list:
        logger.info("Matching curves for turbine %03d..." % ti)
        ref_turb_subset = turbs_ref_list[ti]
        ref_turb_subset = [
            r for r in ref_turb_subset if all(np.isnan(bias_output_list[ti]["T%03d" % r]))
        ]

        for ii, idx_chunk in enumerate(idx_chunks):
            df_subset = df.loc[idx_chunk]

            for ti_ref in ref_turb_subset:
                wd_ref = np.array(df_subset["wd_%03d" % (ti_ref)])
                wd_turb = np.array(df_subset["wd_%03d" % ti])

                wd_ref = wrap_360(wd_ref)
                wd_turb = wrap_360(wd_turb)

                if sum(np.isnan(wd_turb)) / len(wd_turb) < nan_thrshld:
                    dx_opt, J_opt = opt.match_y_curves_by_offset(
                        yref=wd_ref,
                        ytest=wd_turb,
                        angle_wrapping=True,
                    )

                else:
                    dx_opt = np.nan
                bias_output_list[ti].loc[ii, "T%03d" % ti_ref] = dx_opt
                if ti in turbs_ref_list[ti_ref]:
                    bias_output_list[ti_ref].loc[ii, "T%03d" % ti] = -dx_opt

        logger.info(bias_output_list[ti])

    # Find turbines where dx barely changes (low variance)
    turb_is_clean = ["bad" for _ in turbine_list]
    for ti in turbine_list:
        df_out = bias_output_list[ti]
        for ti_ref in turbs_ref_list[ti]:
            # If synced by less than 5 degrees std, then both OK
            if np.nanstd(df_out["T%03d" % ti_ref]) < 5.0:
                turb_is_clean[ti] = "clean"
                turb_is_clean[ti_ref] = "clean"
            else:
                if (turb_is_clean[ti] == "clean") & (turb_is_clean[ti_ref] == "clean"):
                    turb_is_clean[ti] = "disputed"
                    turb_is_clean[ti_ref] = "disputed"

    for ti in turbine_list:
        if turb_is_clean[ti] == "disputed":
            logger.info(
                "Turbine %03d may or may not have jumps in WD measurement calibration. [DISPUTED]"
                % ti
            )
        elif turb_is_clean[ti] == "clean":
            logger.info(
                "Turbine %03d seems to have no jumps in its WD measurement calibration. [CLEAN]"
                % ti
            )
        elif turb_is_clean[ti] == "bad":
            logger.info(
                "Turbine %03d seems to have one or multiple jumps in "
                "its WD measurement calibration. [BAD]" % ti
            )

    if plot_figure:
        # Plot layout and colormap
        fig, ax = plt.subplots(figsize=(14, 5))
        for ti in turbine_list:
            if turb_is_clean[ti] == "clean":
                clr = "green"
            elif turb_is_clean[ti] == "bad":
                clr = "red"
            elif turb_is_clean[ti] == "disputed":
                clr = "orange"

            ax.plot(
                fm.layout_x[ti],
                fm.layout_y[ti],
                "o",
                markersize=15,
                markerfacecolor=clr,
                markeredgewidth=0.0,
            )
            ax.text(
                fm.layout_x[ti] + 100,
                fm.layout_y[ti],
                "T%03d (%s)" % (ti, turb_is_clean[ti]),
                color="black",
            )
        fig.tight_layout()

    return turb_is_clean
