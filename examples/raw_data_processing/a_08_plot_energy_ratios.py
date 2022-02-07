import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris.tools import floris_interface as wfct

from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc.energy_ratio import energy_ratio_suite
from flasc.utilities import printnow as print
from flasc import floris_tools as ftools


def load_floris():
    root_path = os.path.dirname(os.path.abspath(__file__))
    fi = wfct.FlorisInterface(
        os.path.join(root_path, "..", "demo_dataset", "demo_floris_input.json")
    )
    return fi


def load_data():
    # Load the data
    print("Loading .ftr data. This may take a minute or two...")
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_path, "data", "07_wdbias_filtered_data")
    df_scada = pd.read_feather(os.path.join(data_path, "scada_data_60s.ftr"))
    return df_scada


if __name__ == "__main__":
    # This script demonstrates how we can use the postprocessed data to
    # compare the SCADA data to one of the wake models in FLORIS. This
    # script plots the energy ratios for our artificial dataset and
    # compares them with the energy ratio predictions from FLORIS.

    # Specify our test and wd_measurement turbine
    turb_wd_measurement = [0]
    test_turbines = [1]

    # Specify energy ratio settings
    ws_step = 5.0
    wd_step = 3.0
    wd_bin_width = 3.0
    N = 1  # Bootstrapping sample size (higher is better for UQ, but slower)

    # Get data and floris object
    df = load_data()
    fi = load_floris()

    # Plot wind farm layout
    fi.vis_layout()

    # Get dataframe defining which turbines are upstream for what wind dirs
    df_upstream = ftools.get_upstream_turbs_floris(fi)

    # Assign a reference wind direction and wind speed
    print("Processing dataframe: selecting reference wd, ws and pow_ref")
    df = dfm.set_wd_by_turbines(df, turb_wd_measurement)
    df = dfm.set_ws_by_upstream_turbines_in_radius(
        df,
        df_upstream,
        turb_no=test_turbines[0],
        x_turbs=fi.layout_x,
        y_turbs=fi.layout_y,
        max_radius=5000.0,
        include_itself=True,
    )

    # Get FLORIS predictions for SCADA dataframe
    root_path = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(root_path, "df_fi_approx.ftr")
    if os.path.exists(fn):
        df_fi_approx = pd.read_feather(fn)
    else:
        df_fi_approx = ftools.calc_floris_approx_table(
            fi=fi,
            wd_array=np.arange(0.0, 360.0, 3.0),
            ws_array=np.arange(6.0, 10.0, 1.0),
            num_workers=4,
            num_threads=40,
        )
        df_fi_approx.to_feather(fn)

    df_fi = ftools.interpolate_floris_from_df_approx(
        df=df, df_approx=df_fi_approx, method="linear", verbose=True
    )

    # Set reference power for both our SCADA data and for our FLORIS data
    df = dfm.set_pow_ref_by_upstream_turbines_in_radius(
        df,
        df_upstream,
        turb_no=test_turbines[0],
        x_turbs=fi.layout_x,
        y_turbs=fi.layout_y,
        max_radius=5000.0,
        include_itself=True,
    )

    df_fi = dfm.set_pow_ref_by_upstream_turbines_in_radius(
        df_fi,
        df_upstream,
        turb_no=test_turbines[0],
        x_turbs=fi.layout_x,
        y_turbs=fi.layout_y,
        max_radius=5000.0,
        include_itself=True,
    )

    # Calculate and plot energy ratios
    s = energy_ratio_suite.energy_ratio_suite(verbose=False)
    s.add_df(df, "SCADA data (wind direction uncalibrated)")
    s.add_df(df_fi, "FLORIS")

    print("Calculating energy ratios with bootstrapping (N={}).".format(N))
    print("This may take a couple seconds...")
    s.set_masks(ws_range=(6.0, 10.0))
    s.get_energy_ratios(
        test_turbines=test_turbines,
        wd_step=wd_step,
        ws_step=ws_step,
        wd_bin_width=wd_bin_width,
        N=N,
        percentiles=[5.0, 95.0],
        verbose=True,
    )
    ax = s.plot_energy_ratios()
    ax[0].set_title("Energy ratios; test_turbines = {}".format(test_turbines))
    plt.tight_layout()

    plt.show()
