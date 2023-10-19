import unittest

import pandas as pd

from flasc.energy_ratio import total_uplift as tup
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

        er_in = EnergyRatioInput([df_base, df_wake_steering],
                                 ['baseline', 'wake_steering'],
                                 num_blocks=1)

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


        # Check assertions
        self.assertAlmostEqual(total_uplift_result['uplift']['energy_uplift_ctr'],
            18615, places=4) 
        self.assertAlmostEqual(total_uplift_result['uplift']['energy_uplift_ctr_pc'],
            47.22222222, places=4) 


    def test_total_uplift_bootstrap(self):

        # Test the ability to compute the total uplift in energy production with bootstrapping
        # Confirm the "central" answer is deterministic and that the upper and lower bounds
        # make sense
        
        # This time use ratios that are all 1 in the baseline case and between 1.5 and 2.5
        df_base = pd.DataFrame({'wd': [270, 270.,270., 270.,270.,270.],
                           'ws': [7.,7.,7., 8.,8.,8.],
                           'pow_000': [1., 1., 1., 1.,1.,1.],
                           'pow_001': [1., 1., 1., 1.,1., 1.],
        })

        # Define an uplift between 25 and 75%
        df_wake_steering  = pd.DataFrame({'wd': [270, 270., 270.,270.,270.,270.],
                           'ws': [7., 7.,7., 8.,8.,8.],
                           'pow_000': [1., 1., 1., 1.,1.,1.],
                           'pow_001': [1.25, 1.25,1.25, 1.75, 1.75 , 1.75],
        })

        er_in = EnergyRatioInput([df_base, df_wake_steering],
                                 ['baseline', 'wake_steering'],
                                   num_blocks=df_base.shape[0])

        total_uplift_result_1 = tup.compute_total_uplift(
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
            uplift_names = ['uplift'],
            N=10
        )

        total_uplift_result_2 = tup.compute_total_uplift(
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
            uplift_names = ['uplift'],
            N=10
        )
        
        # Confirm determinism of the central
        self.assertAlmostEqual(total_uplift_result_1['uplift']['energy_uplift_ctr_pc'], 
                                total_uplift_result_2['uplift']['energy_uplift_ctr_pc'], 
                                places=4) 

        # Check accuraccy of centreal result
        self.assertAlmostEqual(total_uplift_result_1['uplift']['energy_uplift_ctr_pc'],  
                               50.0, 
                               places=4) 

        # Check reasonableness of upper/lower bounds
        self.assertGreaterEqual(total_uplift_result_1['uplift']['energy_uplift_lb_pc'],  25.0) 
        self.assertLessEqual(total_uplift_result_1['uplift']['energy_uplift_ub_pc'],  75.0) 
