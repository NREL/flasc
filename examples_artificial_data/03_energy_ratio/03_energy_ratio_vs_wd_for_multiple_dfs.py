import os

import matplotlib.pyplot as plt
import pandas as pd
from floris.utilities import wrap_360

from flasc.analysis import energy_ratio as er
from flasc.analysis.energy_ratio_input import EnergyRatioInput
from flasc.data_processing import dataframe_manipulations as dfm
from flasc.utilities import floris_tools as fsatools
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
    # Load data and floris object
    df = load_data()
    fi, _ = load_floris()

    # Visualize layout
    fig, ax = plt.subplots()
    ax.plot(fm.layout_x, fm.layout_y, "ko")
    for ti in range(len(fm.layout_x)):
        ax.text(fm.layout_x[ti], fm.layout_y[ti], "T{:02d}".format(ti))
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x-direction (m)")
    ax.set_ylabel("y-direction (m)")

    # We first need to define a wd against which we plot the energy ratios
    # In this example, we set the wind direction to be equal to the mean
    # wind direction between all turbines
    df = dfm.set_wd_by_all_turbines(df)

    # We also need to define a reference wind speed and a reference power
    # production against to normalize the energy ratios with. In this
    # example, we set the wind speed equal to the mean wind speed
    # of all upstream turbines. The upstream turbines are automatically
    # derived from the turbine layout and the wind direction signal in
    # the dataframe, df['wd']. The reference power production is set
    # as the average power production of all upstream turbines.
    df_upstream = fsatools.get_upstream_turbs_floris(fi, wd_step=5.0)
    df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
    df = dfm.set_pow_ref_by_upstream_turbines(df, df_upstream)

    # Now we generate a copy of the original dataframe and shift the
    # reference wind direction measurement upward by 7.5 degrees.
    df2 = df.copy()
    df2["wd"] = wrap_360(df2["wd"] + 7.5)

    # Initialize the energy ratio input object and add dataframes
    # separately. We will add the original data and the manipulated
    # dataset.
    er_in = EnergyRatioInput([df, df2], ["Original data", "Data with wd bias of 7.5 degrees"])

    # Calculate the energy ratios for test_turbines = [1] for a subset of
    # wind directions with uncertainty quantification using 50 bootstrap
    # samples
    er_out = er.compute_energy_ratio(
        er_in,
        test_turbines=[1],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=4.0,
        wd_min=20.0,
        wd_max=90.0,
        N=50,
        percentiles=[5.0, 95.0],
    )
    er_out.plot_energy_ratios()

    # Look at another test turbine with the same masked datasets
    er_out = er.compute_energy_ratio(
        er_in,
        test_turbines=[3],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=4.0,
        wd_min=20.0,
        wd_max=90.0,
        N=50,
        percentiles=[5.0, 95.0],
    )
    er_out.plot_energy_ratios()
    er_out.plot_energy_ratios(polar_plot=True)  # Also show in a polar plot
    plt.show()
