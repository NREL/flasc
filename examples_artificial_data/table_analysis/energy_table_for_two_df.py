import os
import pandas as pd
import numpy as np

from floris import tools as wfct
from floris.utilities import wrap_360
from flasc.energy_ratio import energy_ratio_suite
from flasc.dataframe_operations import (
    dataframe_manipulations as dfm,
)
from flasc import floris_tools as fsatools


def load_data():
    # Load dataframe with scada data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(
        root_dir, "..", "demo_dataset", "demo_dataset_scada_60s.ftr"
    )
    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            "Please run ./examples_artificial_data/demo_dataset/"
            + "generate_demo_dataset.py before try"
            + "ing any of the other examples."
        )
    df = pd.read_feather(ftr_path)
    return df


def load_floris():
    # Initialize the FLORIS interface fi
    print("Initializing the FLORIS object for our demo wind farm")
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(
        file_path, "../demo_dataset/demo_floris_input.json"
    )
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


if __name__ == "__main__":
    # Set a random seed
    np.random.seed(0)

    # Load data and FLORIS
    df = load_data()
    fi = load_floris()

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
    # as the average power production of turbines 0 and 6, which are
    # always upstream for wind directions between 20 and 90 deg.
    df_upstream = fsatools.get_upstream_turbs_floris(fi)

    # Set the wind speed, power and ti using the upstream turbines
    df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
    df = dfm.set_pow_ref_by_upstream_turbines(df, df_upstream)
    df = dfm.set_ti_by_upstream_turbines(df, df_upstream)

    # Make a second dataframe with some random noise applied to wind direction
    df2 = df.copy()
    df2["wd"] = df2["wd"] + np.random.normal(0, 1.0, df2.shape[0])
    df2["wd"] = wrap_360(df2["wd"])

    # Also make a gap between 8 and 10 m/s
    df2 = df2[(df2.ws < 8) | (df2.ws > 10)]
    # df2['ws'] = df2['ws'] + np.random.normal(0,0.5,df2.ws.values.shape)

    # Initialize the energy ratio suite object and add each dataframe
    # separately. We will import the original data and the manipulated
    # dataset.
    fsc = energy_ratio_suite.energy_ratio_suite()
    fsc.add_df(df, "baseline")
    fsc.add_df(df2, "random_wd_perturbation")

    # Print the dataframes to see if everything is imported properly
    fsc.print_dfs()
    fsc.set_masks(ws_range=(5.0, 11.0), wd_range=(40.0, 48.0))

    # Get energy ratios for test_turbine 1 with return_detailed_output=True
    energy_ratios_t1 = fsc.get_energy_ratios(
        test_turbines=[1],
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=2.0,
        N=1,
        percentiles=[5.0, 95.0],
        return_detailed_output=True,
        verbose=True
    )

    # Generate table
    root_path = os.path.dirname(os.path.abspath(__file__))
    fsc.export_detailed_energy_info_to_xlsx(
        fout_xlsx=os.path.join(root_path, "energy_table_t1.xlsx"),
        fi=fi,
    )
