from io import StringIO
import os
import pandas as pd
import polars as pl
import numpy as np
import pytest

import unittest

from floris import tools as wfct
from flasc.dataframe_operations import dataframe_manipulations as dfm
from flasc import floris_tools as ftools
from flasc.utilities_examples import load_floris_artificial as load_floris
from flasc.energy_ratio import energy_ratio as erp
from flasc.energy_ratio import total_uplift as tup
from flasc.energy_ratio.energy_ratio_utilities import add_reflected_rows
from flasc.energy_ratio.energy_ratio_input import EnergyRatioInput



class TestTotalUplift(unittest.TestCase):

    def test_total_uplift(self):

        # Test the ability to compute the total uplift in energy production
        
        # Test the returned energy ratio assuming alternative weightings of the wind speed bins
        df_base = pd.DataFrame({'wd': [270, 270., 270.,270.,272.],
                           'ws': [7., 8., 8.,8.,8.],
                           'pow_000': [1., 1., 1., 1.,10.],
                           'pow_001': [1., 1., 1., 1.,10.],
        })

        df_wake_steering  = pd.DataFrame({'wd': [270, 270., 270.,270.,272.],
                           'ws': [7., 7., 8.,8.,8.],
                           'pow_000': [1., 1., 1., 1.,20.],
                           'pow_001': [2., 2., 1., 1.,30.],
        })

        er_in = EnergyRatioInput([df_base, df_wake_steering],['baseline', 'wake_steering'], num_blocks=1)

        total_uplift_result = tup.compute_total_uplift(
            er_in,
            ref_turbines=[0],
            test_turbines=[1],
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_min = 269.,
            wd_step=2.0,
            ws_min = 0.5, # Make sure bin labels land on whole numbers
            weight_by='min',
            uplift_pairs = ['baseline', 'wake_steering'],
            uplift_names = ['uplift']
        )

        # Total energy production is computed via a weighted sum over differences in power ratio
        # Delta Power Ratio
        # Bin 7m/s, 270 deg: 
        #  R Base: 1/1
        #  R Control: 2 /1
        #  Delta R: 2 - 1 = 1
        #  P_ref: 1
        #  count: (weight by min) 1
        # Bin 8 m/s, 270 deg:
        #  R Base: 1/1
        #  R Control: 1 / 1
        #  Delta R: 1 - 1 = 0
        #  P_ref: 1
        #  count: (weight by min) 2
        # Bin 8 m/s, 272 deg:
        #  R Base: 10/10
        #  R Control: 30/20
        #  Delta R: 1.5 - 1 = 0.5
        #  P_ref: 15
        #  count: (weight by min) 1

        # Weights
        # f(7,270) = 1/4
        # f(8, 270) = 1/2
        # f(8, 272) = 1/4

        # Delta_AEP = 8760 * ((1/4 * 1 * 1) + (1/2 * 0 * 1) + (1/4 * 0.5 * 15)) = 18,615
        # Base_AEP = 8760 * ((1/4 * 1 * 1) + (1/2 * 1 * 1) + (1/4 * 1 * 15)) = 39,420
        # Percent Delta AEP = 100 * (18615 / 39420) = 47.22222222


        # Unpack the result and check assertions
        delta_aep, percent_delta_aep = total_uplift_result['uplift']
        self.assertAlmostEqual(delta_aep,  18615  , places=4) 
        self.assertAlmostEqual(percent_delta_aep,  47.22222222  , places=4) 
