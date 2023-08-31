import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris.utilities import wrap_360

from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc import floris_tools as ftools
from flasc.energy_ratio import energy_ratio as er
from flasc.energy_ratio.energy_ratio_input import EnergyRatioInput
from flasc.visualization import plot_floris_layout
from flasc.utilities_examples import load_floris_artificial as load_floris


def load_data():
    # Load dataframe with artificial SCADA data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir, '..', '01_raw_data_processing', 'postprocessed',
        'df_scada_data_600s_filtered_and_northing_calibrated.ftr'
    )
    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            'Please run the scripts in /raw_data_processing/' +
            'before trying any of the other examples.'
        )
    df = pd.read_feather(ftr_path)
    return df


def get_energy_ratio(df, ti, aligned_wd):
    # Calculate and plot energy ratios
    #s = energy_ratio_suite.energy_ratio_suite(verbose=False)
    er_in = EnergyRatioInput([df], ['Raw data (wind direction calibrated)'])
    #s.add_df(df, 'Raw data (wind direction calibrated)')
    return er.compute_energy_ratio(
            er_in,
            test_turbines=[ti],
            use_predefined_ref=True,
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_step=15.0,
            wd_bin_overlap_radius=0.0,
            wd_min=aligned_wd-15.0/2,
            wd_max=aligned_wd+15.0/2,
            ws_min=6.0,
            ws_max=10.0,
            N=10,
            percentiles=[5.0, 95.0]
        )


def _process_single_wd(wd, wd_bin_width, turb_wd_measurement, df_upstream, df):
    # In this function, we calculate the energy ratios of all upstream
    # turbines for a single wind direction bin and single wind speed bin.
    # The difference in energy ratios between different upstream turbines
    # gives a strong indication of the heterogeneity in the inflow wind
    # speeds for that mean inflow wind direction.
    print("Processing wind direction = {:.1f} deg.".format(wd))
    wd_bins = [[wd - wd_bin_width / 2.0, wd + wd_bin_width / 2.0]]

    # Determine which turbines are upstream
    if wd > df_upstream.iloc[0]["wd_max"]:
        turbine_array = df_upstream.loc[
            (wd > df_upstream["wd_min"]) & (wd <= df_upstream["wd_max"]),
            "turbines"
        ].values[0]

    # deal with wd = 0 deg (or close to 0.0)
    else:
        turbine_array = df_upstream.loc[
            (wrap_360(wd + 180) > wrap_360(df_upstream["wd_min"] + 180.0)) &
            (wrap_360(wd + 180) <= wrap_360(df_upstream["wd_max"] + 180)),
            "turbines"
        ].values[0]

    # Load data and limit region
    df = df.copy()
    pow_cols = ["pow_{:03d}".format(t) for t in turbine_array]
    df = df.dropna(subset=pow_cols)

    # Filter dataframe and set a reference wd and ws
    df = dfm.set_wd_by_turbines(df, turb_wd_measurement)
    df = dfm.filter_df_by_wd(df, [wd - wd_bin_width, wd + wd_bin_width])
    df = dfm.set_ws_by_turbines(df, turbine_array)
    df = dfm.filter_df_by_ws(df, [6, 10])

    # Set reference power for df and df_fi as the average power
    # of all upstream turbines
    df = dfm.set_pow_ref_by_turbines(df, turbine_array)

    results_scada = []
    for ti in turbine_array:
        # Get energy ratios
        er_out = get_energy_ratio(df, ti, wd)
        results_scada.append(er_out.df_result)

    results_scada = pd.concat(results_scada)
    energy_ratios = np.array(results_scada["Raw data (wind direction calibrated)"], dtype=float)
    energy_ratios_lb = np.array(results_scada["Raw data (wind direction calibrated)_lb"], dtype=float)
    energy_ratios_ub = np.array(results_scada["Raw data (wind direction calibrated)_ub"], dtype=float)

    return pd.DataFrame({
            "wd": [wd],
            "wd_bin_width": [wd_bin_width],
            "upstream_turbines": [turbine_array],
            "energy_ratios": [energy_ratios],
            "energy_ratios_lb": [energy_ratios_lb],
            "energy_ratios_ub": [energy_ratios_ub],
            "ws_ratios": [energy_ratios**(1/3)],
    })


def _plot_single_wd(df):
    fig, ax = plt.subplots()
    turbine_array = df.loc[0, "upstream_turbines"]

    x = range(len(turbine_array))
    ax.fill_between(
        x,
        df.loc[0, "energy_ratios_lb"],
        df.loc[0, "energy_ratios_ub"],
        color="k",
        alpha=0.30
    )
    ax.plot(x, df.loc[0, "energy_ratios"], "-o", color='k', label="SCADA")
    ax.grid(True)
    ax.set_xticks(x)
    ax.set_xticklabels(["T{:03d}".format(t) for t in turbine_array])
    ax.set_ylabel("Energy ratio of upstream turbines w.r.t. the average (-)")
    ax.set_title("Wind direction = {:.2f} deg.".format(df.loc[0, "wd"]))
    ax.set_ylim([0.85, 1.20])
    return fig, ax


if __name__ == "__main__":
    # Load FLORIS and plot the layout
    fi, _ = load_floris()
    plot_floris_layout(fi, plot_terrain=False)

    # Load the SCADA data
    df_full = load_data()

    # Now specify which turbines we want to use in the analysis. Basically,
    # we want to use all the turbines besides the ones that we know have
    # an unreliable wind direction measurement. Here, for explanation purposes,
    # we just exclude turbine 3 from our analysis.
    nturbs = len(fi.layout_x)
    bad_turbs = [3]  # Just hypothetical situation: assume turbine 3 gave faulty wind directions so we ignore it
    turb_wd_measurement = [i for i in range(nturbs) if i not in bad_turbs]

    # We use a wind direction bin width of 15 deg. Thus, if we look at
    # heterogeneity with winds coming from the west (270 deg), then we
    # use all data reporting a wind direction measurement between 262.5
    # and 277.5 deg, when we have a wd_bin_width of 15.0 deg.
    wd_bin_width = 15.0

    # Now calculate which turbines are upstream and for what wind directions,
    # using a very simplified model as part of FLASC.
    df_upstream = ftools.get_upstream_turbs_floris(fi, wake_slope=0.3)

    # Finally, for various wind directions, calculate the energy ratios of
    # all upstream turbines. That gives a good idea of the heterogeneity
    # in the inflow wind speeds. Namely, turbines that consistently see
    # a higher energy ratio, also likely consistently see a higher wind speed.
    df_list = []
    for wd in np.arange(0.0, 360.0, 15.0):
        df = _process_single_wd(wd, wd_bin_width, turb_wd_measurement, df_upstream, df_full)
        fig, ax  = _plot_single_wd(df)  # Plot the results
        df_list.append(df)

    # Finally merge the results to a single dataframe and print
    df = pd.concat(df_list).reset_index(drop=True)
    print(df)

    plt.show()