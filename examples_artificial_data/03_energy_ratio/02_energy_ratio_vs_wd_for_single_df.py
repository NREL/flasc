import os

import matplotlib.pyplot as plt
import pandas as pd

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
    fm, _ = load_floris()

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

    # We reduce the dataframe to only data where the wind direction
    # is between 20 and 90 degrees.
    df = dfm.filter_df_by_wd(df=df, wd_range=[20.0, 90.0])
    df = df.reset_index(drop=True)

    # We also need to define a reference wind speed and a reference power
    # production against to normalize the energy ratios with. In this
    # example, we set the wind speed equal to the mean wind speed
    # of all upstream turbines. The upstream turbines are automatically
    # derived from the turbine layout and the wind direction signal in
    # the dataframe, df['wd']. The reference power production is set
    # as the average power production of turbines 0 and 6, which are
    # always upstream for wind directions between 20 and 90 deg.
    df_upstream = fsatools.get_upstream_turbs_floris(fm)
    df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
    df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

    # # Initialize energy ratio object for the dataframe
    er_in = EnergyRatioInput([df], ["baseline"])

    # Get energy ratio without uncertainty quantification
    er_out = er.compute_energy_ratio(
        er_in,
        test_turbines=[1],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=1.0,
    )
    ax = er_out.plot_energy_ratios()
    ax[0].set_title("Energy ratios for turbine 001 without UQ")
    plt.tight_layout()

    # Also show polar plot
    ax = er_out.plot_energy_ratios(polar_plot=True, show_wind_speed_distribution=False)
    ax[0].set_title("Energy ratios for turbine 001 without UQ")
    plt.tight_layout()

    # Get energy ratio with uncertainty quantification
    # using N=20 bootstrap samples and 5-95 percent conf. bounds.
    er_out = er.compute_energy_ratio(
        er_in,
        test_turbines=[1],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=1.0,
        N=20,
        percentiles=[5.0, 95.0],
    )
    ax = er_out.plot_energy_ratios()
    ax[0].set_title("Energy ratios for turbine 001 with UQ " + "(N=20, 90% confidence interval)")
    plt.tight_layout()

    # Get energy ratio with uncertainty quantification
    # using N=20 bootstrap samples and without block bootstrapping.
    er_in_noblocks = EnergyRatioInput([df], ["baseline"], num_blocks=len(df))
    er_out = er.compute_energy_ratio(
        er_in_noblocks,
        test_turbines=[1],
        use_predefined_ref=True,
        use_predefined_wd=True,
        use_predefined_ws=True,
        wd_step=2.0,
        ws_step=1.0,
        N=20,
        percentiles=[5.0, 95.0],
    )
    ax = er_out.plot_energy_ratios()
    ax[0].set_title(
        "Energy ratios for turbine 001 with UQ " + "(N=20, Normal (not Block) Bootstrapping)"
    )
    plt.tight_layout()

    plt.show()
