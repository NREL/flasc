from io import StringIO
import os
import numpy as np
import pandas as pd

import unittest

from floris import tools as wfct
from flasc.energy_ratio import energy_ratio
from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc import floris_tools as ftools


def load_floris():
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "..", "examples", "demo_dataset", "demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


def load_data():
    # 4-line data file
    csv = (
        "time,wd_000,wd_001,wd_002,wd_003,wd_004,wd_005,wd_006," +
        "ws_000,ws_001,ws_002,ws_003,ws_004,ws_005,ws_006,pow_000," +
        "pow_001,pow_002,pow_003,pow_004,pow_005,pow_006\n" +
        "2019-01-01 00:38:00+00:00,162.456,166.355,156.422,129.406," +
        "133.190,141.612,143.815,2.371,2.120,1.733,1.951,2.064,2.102," +
        "2.047,2.219,2.074,2.228,1.843,2.263,1.840,2.110\n" +
        "2019-01-01 00:47:00+00:00,148.948,165.333,158.185,121.214," +
        "134.470,143.641,143.457,2.160,2.325,2.197,2.144,1.688,1.851," +
        "2.341,4.469,3.124,4.190,2.580,4.319,2.639,4.222\n" +
        "2019-01-01 00:51:00+00:00,154.352,171.207,160.779,118.388," +
        "133.731,144.043,144.885,1.798,2.002,1.736,2.087,1.497,1.978," +
        "1.883,0.000,0.000,0.000,0.000,0.000,0.000,0.000\n" +
        "2019-01-01 01:04:00+00:00,152.255,163.902,155.395,122.070," +
        "136.340,145.449,149.110,1.882,1.397,2.009,1.738,1.778,1.052," +
        "1.556,0.000,0.000,0.000,0.000,0.000,0.000,0.000\n"
    )
    f = StringIO(csv)
    return pd.read_csv(f)


class TestEnergyRatio(unittest.TestCase):
    def test_energy_ratio_regression(self):
        # Load data and FLORIS model
        fi = load_floris()
        df = load_data()
        df = dfm.set_wd_by_all_turbines(df)
        df_upstream = ftools.get_upstream_turbs_floris(fi)
        df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
        df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

        # Get energy ratios
        era = energy_ratio.energy_ratio(df_in=df, verbose=True)
        out = era.get_energy_ratio(
            test_turbines=[1],
            wd_step=2.0,
            ws_step=1.0,
            wd_bin_width=3.0,
        )

        self.assertAlmostEqual(out.loc[0, "baseline"], 0.718904, places=4)
        self.assertAlmostEqual(out.loc[1, "baseline"], 0.958189, places=4)
        self.assertAlmostEqual(out.loc[2, "baseline"], 0.958189, places=4)

        self.assertEqual(out.loc[0, "bin_count"], 2)
        self.assertEqual(out.loc[1, "bin_count"], 3)
        self.assertEqual(out.loc[2, "bin_count"], 1)
