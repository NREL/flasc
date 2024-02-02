import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flasc.analysis.energy_ratio_heterogeneity_mapper import heterogeneity_mapper
from flasc.preprocessing import dataframe_manipulations as dfm
from flasc.utilities import floris_tools as ftools

# from flasc.visualization import plot_floris_layout
from flasc.utilities.utilities_examples import load_floris_artificial as load_floris


def load_data():
    # Load dataframe with artificial SCADA data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir,
        "..",
        "01_raw_data_processing",
        "postprocessed",
        "df_scada_data_600s_filtered_and_northing_calibrated.ftr",
    )
    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            "Please run the scripts in /raw_data_processing/"
            + "before trying any of the other examples."
        )
    df = pd.read_feather(ftr_path)
    return df


if __name__ == "__main__":
    # Load FLORIS and plot the layout
    fi, _ = load_floris()

    # Now specify which turbines we want to use in the analysis. Basically,
    # we want to use all the turbines besides the ones that we know have
    # an unreliable wind direction measurement. Here, for explanation purposes,
    # we just exclude turbine 3 from our analysis.
    nturbs = len(fi.layout_x)
    bad_turbs = [
        3
    ]  # Just hypothetical situation: assume turbine 3 gave faulty wind directions so we ignore it
    turb_wd_measurement = [i for i in range(nturbs) if i not in bad_turbs]

    # Load the SCADA data and assign the freestream wind direction
    df_full = load_data()
    df_full = dfm.set_wd_by_turbines(df_full, turb_wd_measurement)

    # We use a wind direction bin width of 15 deg. Thus, if we look at
    # heterogeneity with winds coming from the west (270 deg), then we
    # use all data reporting a wind direction measurement between 262.5
    # and 277.5 deg, when we have a wd_bin_width of 15.0 deg.
    wd_bin_width = 15.0

    # Now calculate which turbines are upstream and for what wind directions,
    # using a very simplified model as part of FLASC. We use a wide wake
    # slope since we use a large value for wd_bin_width too.
    df_upstream = ftools.get_upstream_turbs_floris(fi, wake_slope=0.70)

    # Load the FLASC heterogeneity mapper
    hm = heterogeneity_mapper(df_raw=df_full, fi=fi)

    # For all wind directions from 0 to 360 deg, calculate the energy ratios of
    # all upstream turbines. That gives a good idea of the heterogeneity
    # in the inflow wind speeds. Namely, turbines that consistently see
    # a higher energy ratio, also likely consistently see a higher wind speed.
    df_heterogeneity = hm.estimate_heterogeneity(
        df_upstream=df_upstream,
        wd_bin_width=wd_bin_width,
        wd_array=np.arange(0.0, 360.0, 30.0),
        ws_range=[6.0, 11.0],
    )
    print("df_heterogeneity:")
    print(df_heterogeneity)

    # Extract a FLORIS heterogeneity map
    df_fi_hetmap = hm.generate_floris_hetmap()
    print("")
    print("df_fi_map:")
    print(df_fi_hetmap)

    # Generate a heterogeneity contour plot over the turbine layout plot
    root_path = os.path.dirname(os.path.abspath(__file__))
    pdf_save_path = os.path.join(root_path, "heterogeneity_layouts.pdf")
    hm.plot_layout(plot_background_flow=True, ylim=[0.90, 1.10], pdf_save_path=pdf_save_path)

    # Plot individual graphs to showcase heterogeneity in detail
    hm.plot_graphs()
    plt.show()
