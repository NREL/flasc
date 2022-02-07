import numpy as np
import pandas as pd
from scipy.stats import circmean

import unittest

from flasc.dataframe_operations import dataframe_manipulations as dfm


def load_data():
    # Create a demo dataframe
    N = 100
    df_full = pd.DataFrame()
    wd_array = np.array([350, 3., 8., 5.])
    ws_array = np.array([5., 17., 0., 29.])
    ti_array = np.array([0.03, 0.09, 0.25, 0.30])
    pow_array = np.array([1500., 1800., 3500., 50.])

    for ti in range(len(wd_array)):
        df_full['wd_%03d' % ti] = np.repeat(wd_array[ti], N)
        df_full['ws_%03d' % ti] = np.repeat(ws_array[ti], N)
        df_full['ti_%03d' % ti] = np.repeat(ti_array[ti], N)
        df_full['pow_%03d' % ti] = np.repeat(pow_array[ti], N)

    return df_full


def get_df_upstream():
    df_upstream = pd.DataFrame({'wd_min': [0., 180.],
                                'wd_max': [180., 360.],
                                'turbines': [[0, 1], [2, 3]]})
    return df_upstream


class TestDataframeManipulations(unittest.TestCase):
    def test_set_by_all(self):
        df_test = load_data().copy()
        df_test = dfm.set_wd_by_all_turbines(df_test)
        df_test = dfm.set_ws_by_all_turbines(df_test)
        df_test = dfm.set_ti_by_all_turbines(df_test)

        wd_ref = circmean([350, 3., 8., 5.], high=360.)
        ws_ref = np.mean([5., 17., 0., 29.])
        ti_ref = np.mean([0.03, 0.09, 0.25, 0.30])

        self.assertAlmostEqual(wd_ref, df_test.loc[0, 'wd'])
        self.assertAlmostEqual(ws_ref, df_test.loc[0, 'ws'])
        self.assertAlmostEqual(ti_ref, df_test.loc[0, 'ti'])

    def test_set_by_turbines(self):
        # Test set_*_by_turbines functions
        df_test = load_data().copy()
        turbine_list = [0, 2]
        df_test = dfm.set_wd_by_turbines(df_test, turbine_numbers=turbine_list)
        df_test = dfm.set_ws_by_turbines(df_test, turbine_numbers=turbine_list)
        df_test = dfm.set_ti_by_turbines(df_test, turbine_numbers=turbine_list)

        self.assertAlmostEqual(df_test.loc[0, 'wd'], circmean([350., 8.], high=360.))
        self.assertAlmostEqual(df_test.loc[0, 'ws'], np.mean([5., 0.]))
        self.assertAlmostEqual(df_test.loc[0, 'ti'], np.mean([0.03, 0.25]))

    def test_set_by_upstream_turbines(self):
        # Test set_*_by_upstream_turbines functions
        df_test = load_data().copy()
        df_upstream = get_df_upstream()

        df_test = dfm.set_wd_by_all_turbines(df_test)
        df_test = dfm.set_ws_by_upstream_turbines(df_test, df_upstream)
        df_test = dfm.set_ti_by_upstream_turbines(df_test, df_upstream)

        self.assertAlmostEqual(df_test.loc[0, 'ws'], np.mean([5., 17.]))
        self.assertAlmostEqual(df_test.loc[0, 'ti'], np.mean([0.03, 0.09]))

    def test_set_by_upstream_turbines_in_radius(self):
        # Test set_*_by_upstream_turbines_in_radius functions
        df_test = load_data().copy()
        df_upstream = get_df_upstream()

        df_test = dfm.set_wd_by_all_turbines(df_test)
        df_test = dfm.set_ws_by_upstream_turbines_in_radius(
            df_test,
            df_upstream,
            turb_no=0,
            x_turbs=np.array([0., 500., 1000., 1500.]),
            y_turbs=np.array([0., 500., 1000., 1500.]),
            max_radius=1000,
            include_itself=True  # Include itself
        )
        df_test = dfm.set_ti_by_upstream_turbines_in_radius(
            df_test,
            df_upstream,
            turb_no=0,
            x_turbs=np.array([0., 500., 1000., 1500.]),
            y_turbs=np.array([0., 500., 1000., 1500.]),
            max_radius=1000,
            include_itself=False  # Exclude itself
        )
        df_test = dfm.set_pow_ref_by_upstream_turbines_in_radius(
            df_test,
            df_upstream,
            turb_no=0,
            x_turbs=np.array([0., 500., 1000., 1500.]),
            y_turbs=np.array([0., 500., 1000., 1500.]),
            max_radius=1000,
            include_itself=True  # Include itself
        )

        self.assertAlmostEqual(df_test.loc[0, 'ws'], np.mean([5., 17.]))
        self.assertAlmostEqual(df_test.loc[0, 'ti'], np.mean([0.09]))
        self.assertAlmostEqual(df_test.loc[0, 'pow_ref'], np.mean([1500., 1800.]))
