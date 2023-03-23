import numpy as np
import pandas as pd
import os

import unittest

from floris.tools import FlorisInterface
from flasc.dataframe_operations import dataframe_filtering as dff
from flasc import floris_tools as ftools


def load_floris():
    # Initialize the FLORIS interface fi
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../examples/demo_dataset/demo_floris_input.yaml")
    fi = FlorisInterface(fi_path)
    return fi


def load_data():
    # Create tiny subset
    return pd.DataFrame({
        "wd": [255.0],
        "ws": [8.0],
        "ti": [0.07],
        "ws_000": [8.0],
        "ws_001": [8.0],
        "ws_002": [8.0],
        "ws_003": [8.0],
        "ws_004": [8.0],
        "ws_005": [8.0],
        "ws_006": [8.0],
        "pow_000": [2.3e6],
        "pow_001": [2.3e6],
        "pow_002": [2.3e6],
        "pow_003": [2.3e6],
        "pow_004": [2.3e6],
        "pow_005": [2.3e6],
        "pow_006": [2.3e6],
    })


class TestDataFrameFiltering(unittest.TestCase):
    def test_impacting_filtering(self):

        # Read file and load FLORIS
        fi = load_floris()
        num_turbs = len(fi.layout_x)

        # Determine which turbines impact which other turbines through their wakes
        df_impacting_turbines = ftools.get_all_impacting_turbines(fi_in=fi, change_threshold=0.01)

        # Create tiny subset
        df_base = load_data()

        # Try a couple scenarios. 1st scenario: T04 waking T03 and T04 is NaN
        # print("\n\n Creating scenario where T4 wakes T3 and T6 and T4 is faulty:")
        df = df_base.copy()
        df["wd"] = 275.0
        df["pow_004"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_003", "pow_004", "pow_006"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_000", "pow_001", "pow_002", "pow_005"]].isna().any().any())  # Non-NaN
                    
        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1 and T1 is faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_001"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_001"]].isna().all().all())
        self.assertTrue(~df[["pow_000", "pow_002", "pow_003", "pow_004", "pow_005", "pow_006"]].isna().any().any())
    
        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1 and T1 and T5 are faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_001"] = np.nan
        df["pow_005"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_001", "pow_005"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_000", "pow_002", "pow_003", "pow_004", "pow_006"]].isna().any().any())  # Non-NaN

        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1 and T5 is faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_005"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_001", "pow_005"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_000", "pow_002", "pow_003", "pow_004", "pow_006"]].isna().any().any())  # Non-NaN

        # Another scenario
        # print("\n\n Creating scenario where T5 wakes T1, T06 wakes T0, and T5 and T6 are faulty:")
        df = df_base.copy()
        df["wd"] = 357.0
        df["pow_005"] = np.nan
        df["pow_006"] = np.nan
        for ti in range(num_turbs):
            df = dff.filter_df_by_faulty_impacting_turbines(df=df, ti=ti, df_impacting_turbines=df_impacting_turbines, verbose=False)
        self.assertTrue(df[["pow_000", "pow_001", "pow_005", "pow_006"]].isna().all().all())  # NaN
        self.assertTrue(~df[["pow_002", "pow_003", "pow_004"]].isna().any().any())  # Non-NaN

        # plot_layout_only(fi)
        # plt.show()
