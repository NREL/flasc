import unittest
from io import StringIO

import numpy as np
import pandas as pd
import polars as pl
import pytest

from flasc import FlascDataFrame
from flasc.analysis import energy_ratio as erp
from flasc.analysis.analysis_input import AnalysisInput
from flasc.analysis.energy_ratio_input import EnergyRatioInput
from flasc.data_processing import dataframe_manipulations as dfm
from flasc.utilities import floris_tools as ftools
from flasc.utilities.energy_ratio_utilities import add_reflected_rows
from flasc.utilities.utilities_examples import load_floris_artificial as load_floris

# Disable line too long for this file for csv block
# ruff: noqa: E501


def load_data():
    # 4-line data file
    csv = """
        time,wd_000,wd_001,wd_002,wd_003,wd_004,wd_005,wd_006,ws_000,ws_001,ws_002,ws_003,ws_004,ws_005,ws_006,pow_000,pow_001,pow_002,pow_003,pow_004,pow_005,pow_006
        2019-01-01 06:59:00+00:00,32.465,46.751,31.155,2.771,11.383,18.312,20.249,0.818,0.610,0.677,0.655,0.973,0.925,0.811,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-01 07:00:00+00:00,37.778,43.665,33.640,358.018,10.079,19.140,19.203,0.784,1.152,0.763,1.079,0.907,0.913,0.567,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-01 07:10:00+00:00,32.658,45.328,32.152,0.418,15.204,19.112,20.734,0.845,1.011,0.822,0.991,0.852,1.065,0.908,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-01 07:11:00+00:00,27.256,40.736,33.770,359.602,6.957,20.369,22.872,1.055,0.741,0.846,0.656,0.525,0.548,0.509,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-01 12:37:00+00:00,25.253,47.988,30.970,350.579,10.845,21.061,24.294,0.154,0.595,0.498,0.265,0.016,1.049,0.343,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 05:11:00+00:00,37.893,42.093,32.692,2.519,8.165,26.082,23.578,0.589,0.377,0.015,-0.039,0.094,0.125,0.201,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 05:13:00+00:00,35.941,42.969,37.148,351.097,11.085,22.463,26.485,-0.245,0.274,0.466,0.075,0.129,0.248,0.466,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 14:49:00+00:00,37.741,46.071,39.056,1.354,11.555,21.345,23.444,1.102,1.083,1.015,1.314,1.186,0.817,1.305,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 15:44:00+00:00,29.108,42.570,33.016,352.815,15.322,16.050,19.234,0.986,0.838,0.942,0.802,1.194,0.898,0.884,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 15:45:00+00:00,34.224,47.968,39.128,355.381,11.350,22.841,19.631,1.267,1.760,1.045,1.173,1.287,1.227,1.341,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 15:48:00+00:00,38.906,48.129,34.333,4.807,16.389,21.858,22.396,1.217,0.812,1.316,1.128,1.563,1.195,1.189,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 15:49:00+00:00,31.454,41.641,30.951,357.918,7.845,15.460,23.360,0.981,0.919,0.974,0.732,1.019,1.093,0.828,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 15:51:00+00:00,31.656,49.410,33.623,358.372,14.486,20.613,25.117,0.748,0.989,1.171,1.138,0.764,1.268,1.216,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 15:54:00+00:00,35.553,50.483,39.367,1.258,15.199,25.691,18.470,1.766,1.903,1.528,1.932,1.600,2.022,1.645,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-02 15:59:00+00:00,36.158,46.403,36.629,1.554,16.556,22.269,22.336,1.291,1.081,1.195,1.260,1.495,1.077,0.946,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 00:58:00+00:00,23.458,47.403,33.868,358.290,7.723,18.136,18.613,1.259,0.685,1.319,1.194,1.276,0.781,1.497,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 01:58:00+00:00,36.041,50.614,38.358,4.653,14.333,24.764,21.427,0.455,0.149,0.393,0.132,0.660,0.572,0.669,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 02:07:00+00:00,28.266,45.116,32.174,0.935,14.750,20.792,24.004,0.482,0.587,0.418,0.789,0.377,0.492,0.992,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 02:11:00+00:00,34.119,42.452,33.115,350.271,7.188,15.021,24.265,1.219,0.795,0.530,0.942,0.795,0.955,1.080,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 02:15:00+00:00,32.940,45.611,27.739,351.267,12.439,18.875,18.591,0.821,1.161,1.042,0.970,1.121,1.405,1.304,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 02:22:00+00:00,32.022,48.070,33.792,356.610,12.444,23.168,18.759,1.096,0.944,1.487,1.272,1.249,1.270,1.319,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 02:41:00+00:00,35.061,44.471,34.219,358.169,13.923,23.037,19.358,1.376,1.188,1.471,1.099,1.582,1.400,2.028,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 02:47:00+00:00,39.267,46.410,40.699,0.281,13.175,21.667,25.742,1.371,1.399,1.638,1.414,1.208,1.401,1.292,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 02:51:00+00:00,38.251,49.221,44.095,356.706,13.643,21.180,27.089,1.264,0.998,0.840,1.238,0.744,0.936,0.747,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 08:30:00+00:00,30.964,43.208,37.701,354.289,10.643,22.295,19.222,0.352,0.234,0.334,0.215,0.419,0.506,0.260,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-03 08:32:00+00:00,33.503,48.361,35.274,1.852,14.235,22.660,30.719,0.624,1.177,0.976,0.827,0.672,0.831,0.679,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 09:21:00+00:00,29.182,46.443,36.616,355.550,9.610,22.664,25.734,1.708,1.391,1.444,1.337,1.497,1.750,1.267,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 09:25:00+00:00,31.265,44.955,32.097,1.649,13.451,17.771,13.649,1.697,1.729,1.895,1.772,1.879,2.063,1.808,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 09:42:00+00:00,32.782,42.038,36.107,2.816,15.600,17.952,22.456,2.235,1.743,2.372,2.436,1.819,2.588,2.255,10.981,0.746,9.466,11.600,11.275,11.301,10.915
        2019-01-04 09:43:00+00:00,33.542,40.810,37.943,3.106,6.737,19.853,20.345,1.700,2.178,2.278,2.132,2.298,2.355,1.959,0.647,0.563,0.461,0.662,0.672,0.625,0.636
        2019-01-04 10:06:00+00:00,27.877,42.988,35.453,354.013,8.567,17.277,14.356,0.948,0.144,0.491,0.544,0.624,0.588,0.514,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 10:07:00+00:00,30.733,48.517,35.349,5.763,15.622,24.581,20.936,0.510,0.575,0.764,0.332,0.771,0.818,0.603,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 10:13:00+00:00,31.441,42.164,25.623,359.154,17.344,21.480,26.009,1.523,1.345,1.529,1.599,1.364,1.070,1.797,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 10:17:00+00:00,40.550,45.998,36.389,359.054,14.472,24.968,24.192,1.932,2.157,1.592,2.180,1.781,2.069,1.803,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 10:42:00+00:00,33.788,39.493,35.130,358.278,7.721,16.443,15.269,1.484,0.759,0.828,1.088,0.918,0.936,0.861,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 10:43:00+00:00,29.116,38.175,34.115,354.631,12.744,14.841,20.020,1.753,1.455,1.830,1.590,1.547,1.913,1.575,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 10:50:00+00:00,35.069,41.879,35.504,352.488,11.475,15.875,24.042,1.836,1.526,1.418,1.500,1.393,1.592,1.648,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 10:52:00+00:00,31.989,37.208,29.505,2.014,12.190,19.136,15.278,2.074,1.980,1.866,1.715,2.117,2.013,2.387,1.745,1.256,1.614,1.713,1.740,1.840,1.712
        2019-01-04 10:55:00+00:00,38.496,49.082,41.355,359.156,12.075,30.833,23.172,1.940,2.698,2.250,2.795,3.003,2.449,2.528,21.094,20.133,14.505,21.277,19.828,20.790,20.873
        2019-01-04 10:56:00+00:00,38.538,43.217,33.001,357.001,3.346,17.654,19.695,2.625,2.819,2.631,2.359,2.955,2.657,3.132,24.305,18.381,23.728,25.600,24.869,25.280,25.718
        2019-01-04 10:59:00+00:00,34.904,46.899,34.770,357.340,6.694,22.703,21.247,2.573,2.261,2.157,2.171,2.711,2.429,2.129,17.297,14.515,16.565,18.803,18.741,18.893,18.946
        2019-01-04 11:18:00+00:00,34.945,44.144,36.645,357.853,12.254,21.322,23.941,2.154,2.035,2.565,2.685,2.527,2.799,2.742,16.894,14.407,15.683,17.000,16.682,16.930,16.572
        2019-01-04 14:04:00+00:00,33.509,43.832,33.946,352.447,8.782,18.582,19.701,2.512,2.058,2.675,2.425,2.215,2.617,2.330,10.010,7.662,8.834,10.366,10.236,10.331,10.178
        2019-01-04 14:05:00+00:00,31.264,42.944,35.941,358.323,7.940,23.681,17.239,2.496,2.249,2.007,2.515,2.259,2.110,2.295,9.324,7.855,8.981,9.585,9.894,9.503,9.592
        2019-01-04 14:06:00+00:00,38.935,52.594,37.360,353.533,17.618,23.226,26.067,1.589,2.102,1.160,1.651,1.423,1.313,1.531,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 14:09:00+00:00,41.786,44.052,41.022,2.006,13.765,23.686,27.254,2.996,3.151,2.990,2.737,2.888,3.233,2.811,34.768,32.704,22.694,33.097,36.441,33.473,32.252
        2019-01-04 14:35:00+00:00,39.589,55.658,36.081,4.428,15.868,20.605,22.078,2.956,2.881,2.563,2.778,2.680,2.853,2.634,25.239,24.631,16.148,24.215,24.488,23.136,25.493
        2019-01-04 14:52:00+00:00,34.915,45.367,34.248,1.301,12.060,17.402,16.744,3.274,3.197,2.834,3.121,3.401,3.575,3.016,81.162,72.579,83.099,80.750,81.301,86.889,83.876
        2019-01-04 14:53:00+00:00,32.662,47.322,35.772,356.525,9.982,19.000,23.400,3.576,3.347,3.028,3.172,3.315,3.195,3.503,60.156,54.745,56.173,66.200,63.020,62.092,63.078
        2019-01-04 14:55:00+00:00,37.447,47.331,36.329,359.634,16.036,21.196,21.267,2.927,2.940,2.525,2.421,2.371,2.670,2.636,26.057,24.229,16.034,25.627,25.502,25.392,26.257
        2019-01-04 14:58:00+00:00,32.527,48.092,41.552,1.808,18.346,19.853,19.746,2.266,2.450,2.219,2.443,2.581,2.392,2.570,19.568,19.613,15.394,18.981,19.544,20.165,20.268
        2019-01-04 15:16:00+00:00,34.417,40.769,37.209,358.096,18.116,21.878,20.052,3.443,3.849,3.650,2.924,3.475,3.204,3.070,66.840,67.477,66.767,71.292,65.220,67.315,69.886
        2019-01-04 15:17:00+00:00,36.715,46.488,35.985,2.167,19.283,21.908,21.694,3.153,3.374,3.199,2.838,3.425,3.423,3.481,69.257,63.636,62.549,65.472,68.288,68.896,68.910
        2019-01-04 15:22:00+00:00,32.201,45.764,39.061,2.245,10.290,18.028,21.299,2.879,3.109,3.326,3.409,3.091,3.012,3.040,60.342,58.239,51.217,60.573,63.826,62.250,59.418
        2019-01-04 15:24:00+00:00,31.504,42.481,37.692,357.429,8.748,21.978,22.615,4.245,4.032,3.940,3.709,4.196,4.094,4.137,206.101,184.510,186.720,184.884,202.621,199.131,213.450
        2019-01-04 15:25:00+00:00,28.670,45.920,38.127,1.735,12.317,18.630,25.628,3.905,3.897,3.771,3.894,3.759,4.041,3.774,162.215,151.019,152.475,148.204,153.487,160.474,147.274
        2019-01-04 15:40:00+00:00,38.720,52.999,34.762,2.264,17.447,26.588,20.310,2.006,2.203,2.046,2.347,2.610,2.209,2.144,8.468,8.637,5.259,8.698,8.642,8.344,8.583
        2019-01-04 15:55:00+00:00,35.731,37.342,37.994,353.514,10.399,17.168,14.575,1.597,1.307,1.598,1.648,1.090,1.369,1.409,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 15:56:00+00:00,31.939,45.540,38.801,359.494,10.584,19.964,24.320,1.548,1.259,1.368,1.204,1.298,1.225,1.651,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 16:03:00+00:00,32.200,52.242,39.994,357.316,18.710,27.539,22.928,1.680,1.818,1.705,1.808,1.746,1.997,1.619,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 16:04:00+00:00,39.728,47.031,35.761,3.083,19.624,24.137,21.447,1.986,1.446,1.329,2.066,1.538,1.632,1.730,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 17:03:00+00:00,36.934,42.877,37.482,358.948,10.192,22.815,18.657,1.698,1.749,1.634,1.950,0.933,1.614,1.521,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 18:36:00+00:00,38.604,51.520,39.332,1.295,15.925,20.313,25.051,1.113,0.987,0.996,0.677,0.901,0.911,1.020,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 21:43:00+00:00,35.330,47.381,39.137,356.624,12.039,19.392,22.338,2.349,1.954,1.691,2.071,1.835,1.641,1.833,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-04 23:26:00+00:00,35.908,42.379,35.274,3.078,17.897,24.786,23.936,0.204,0.581,0.556,0.022,0.349,0.538,0.386,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 07:10:00+00:00,33.359,45.035,33.651,356.413,7.391,15.088,20.831,1.132,1.465,1.247,1.597,1.452,1.111,1.677,0.000,0.000,0.000,40.040,0.000,0.000,0.000
        2019-01-05 09:49:00+00:00,40.863,48.691,34.699,0.552,18.153,21.295,21.915,1.834,1.840,1.936,1.795,1.967,2.076,1.823,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 11:09:00+00:00,33.873,44.684,34.454,354.242,15.377,19.295,19.186,0.915,0.517,0.676,1.065,1.029,1.370,1.014,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 11:19:00+00:00,30.614,40.963,38.109,354.828,8.930,16.638,21.913,1.009,0.991,1.292,1.302,1.387,1.192,1.396,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 11:56:00+00:00,37.552,45.994,43.617,359.495,15.607,26.443,20.951,2.265,2.018,2.073,2.072,1.822,2.138,2.095,4.462,4.364,3.384,4.596,4.605,4.581,4.466
        2019-01-05 11:58:00+00:00,33.144,49.995,33.846,1.541,17.417,23.076,20.886,1.038,1.323,1.297,1.227,1.676,1.344,1.486,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 12:53:00+00:00,36.129,46.285,34.734,359.342,15.899,31.010,23.576,2.638,2.771,2.904,2.784,2.945,2.888,2.788,31.165,30.115,21.954,30.488,28.735,30.551,30.765
        2019-01-05 12:54:00+00:00,27.543,44.306,30.138,358.642,7.294,21.550,21.208,3.342,3.329,3.570,3.403,3.231,3.246,3.480,85.214,75.439,77.353,87.211,88.110,86.696,87.244
        2019-01-05 12:57:00+00:00,34.361,38.002,33.342,357.195,9.799,20.230,17.858,2.397,2.264,2.464,2.620,2.336,2.703,2.118,18.874,14.341,17.045,19.325,19.392,19.116,20.692
        2019-01-05 13:03:00+00:00,27.503,45.002,33.197,355.004,12.439,16.955,21.720,3.144,2.913,3.132,2.957,2.786,2.997,3.089,39.501,31.989,36.811,41.180,38.763,40.036,40.955
        2019-01-05 13:04:00+00:00,37.834,42.473,41.474,357.372,18.970,21.877,20.207,2.694,2.894,2.222,2.730,2.892,2.756,2.530,22.779,19.639,17.504,23.900,23.454,23.477,23.857
        2019-01-05 13:07:00+00:00,35.785,48.465,37.985,358.050,16.385,22.071,24.303,2.628,2.305,2.646,2.991,2.621,2.824,2.577,23.268,20.289,17.201,23.009,22.929,22.906,22.959
        2019-01-05 13:08:00+00:00,34.882,51.292,36.314,4.026,9.889,22.303,24.923,2.516,2.310,2.202,2.312,1.800,2.380,2.717,10.019,10.381,7.686,10.308,11.145,9.966,10.269
        2019-01-05 13:09:00+00:00,35.409,46.643,40.521,1.905,13.769,23.994,25.102,1.988,1.749,1.769,1.905,2.005,1.915,2.486,1.711,1.704,1.435,1.703,1.806,1.791,1.812
        2019-01-05 13:12:00+00:00,30.663,46.515,30.792,354.095,10.705,20.358,16.763,2.495,2.125,2.243,1.849,2.278,2.326,2.557,9.553,7.068,8.826,11.095,10.975,10.612,11.459
        2019-01-05 13:15:00+00:00,40.098,53.538,40.925,2.379,13.441,22.386,18.021,2.133,2.267,1.884,2.184,2.009,1.913,1.771,2.105,2.140,1.290,2.185,2.349,2.133,2.201
        2019-01-05 13:18:00+00:00,38.407,46.040,38.232,352.920,15.371,25.557,27.747,2.007,2.062,1.880,1.967,2.302,1.849,2.393,4.767,4.541,4.000,0.413,4.824,4.976,4.702
        2019-01-05 14:40:00+00:00,28.863,48.079,30.674,1.842,13.561,23.249,22.826,2.180,1.963,2.093,2.067,2.178,2.283,1.784,3.423,3.243,3.313,0.068,3.405,3.513,3.505
        2019-01-05 14:55:00+00:00,28.673,45.567,37.180,357.514,13.866,19.292,19.868,2.264,1.606,1.898,2.090,2.015,1.780,1.584,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 15:00:00+00:00,27.488,44.010,33.136,0.491,12.000,18.156,18.492,2.211,2.450,1.842,2.128,2.340,1.896,2.529,8.786,7.199,7.450,8.126,8.264,8.261,8.492
        2019-01-05 15:01:00+00:00,32.578,44.721,36.222,357.821,10.123,23.576,19.156,2.219,2.423,2.252,2.400,2.172,2.370,2.367,16.204,13.615,15.529,16.785,17.245,17.184,17.653
        2019-01-05 15:07:00+00:00,33.055,39.561,29.244,0.914,9.597,16.358,17.136,2.112,1.949,1.641,1.712,1.796,1.720,1.725,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 15:08:00+00:00,26.615,41.587,37.926,352.223,10.859,21.842,20.450,1.792,1.593,1.764,2.211,1.742,2.180,1.619,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 15:13:00+00:00,43.679,48.141,40.680,359.343,11.992,23.733,15.671,1.952,2.289,2.083,2.037,1.945,2.338,2.339,2.675,2.421,2.315,2.557,2.737,2.532,2.622
        2019-01-05 15:36:00+00:00,34.381,46.069,32.921,352.890,10.910,24.587,17.564,1.298,1.577,1.948,1.140,1.468,1.605,1.534,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 15:37:00+00:00,38.429,49.278,37.235,359.613,14.826,23.236,22.771,1.488,1.482,1.450,1.448,1.329,1.585,1.558,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 15:54:00+00:00,31.828,42.681,37.554,357.675,13.417,15.983,16.296,1.672,1.769,1.498,1.790,1.269,1.866,1.419,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 16:05:00+00:00,31.417,47.123,30.183,352.381,10.863,20.446,22.790,1.344,1.111,0.828,1.029,0.864,1.175,0.965,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 16:39:00+00:00,32.941,40.968,29.032,353.423,12.905,21.617,15.210,1.159,0.979,1.167,1.242,1.582,1.330,1.313,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 16:53:00+00:00,26.472,41.095,43.067,348.346,11.003,14.493,19.711,0.999,0.680,1.132,0.665,0.671,0.593,0.718,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 16:59:00+00:00,38.320,46.487,36.815,1.353,18.834,21.010,28.060,0.158,0.501,0.301,0.453,0.433,0.195,0.522,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 18:08:00+00:00,30.944,44.852,33.486,358.552,9.428,22.058,16.047,1.448,1.463,1.458,1.284,1.344,1.715,1.358,0.000,0.000,0.000,0.000,0.000,0.000,0.000
        2019-01-05 19:11:00+00:00,33.416,50.471,35.939,3.520,13.352,20.528,22.078,3.189,2.997,2.760,2.637,2.756,2.726,2.737,26.512,25.636,24.691,27.299,25.907,25.983,27.206
        2019-01-05 19:14:00+00:00,36.273,50.960,43.446,2.367,15.346,22.793,22.742,3.051,3.196,2.552,2.894,3.103,2.907,2.913,33.409,29.525,18.551,31.141,32.227,32.362,33.018
        2019-01-05 19:15:00+00:00,40.076,50.906,40.332,2.129,16.490,23.061,21.499,2.901,3.270,2.661,2.866,2.833,2.570,2.829,33.056,33.250,22.273,31.586,31.987,32.132,34.229
    """
    f = StringIO(csv)
    return pd.read_csv(f)


class TestEnergyRatio(unittest.TestCase):
    def test_energy_ratio_regression(self):
        # Load data and FLORIS model
        fm, _ = load_floris()
        df = load_data()
        df = dfm.set_wd_by_all_turbines(df)
        df_upstream = ftools.get_upstream_turbs_floris(fm)
        df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
        df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

        wd_step = 2.0
        ws_step = 1.0

        a_in = AnalysisInput([df], ["baseline"])

        er_out = erp.compute_energy_ratio(
            a_in,
            ["baseline"],
            test_turbines=[1],
            use_predefined_ref=True,
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_max=360.0,
            wd_min=0.0,
            wd_step=wd_step,
            ws_max=30.0,
            ws_min=0.0,
            ws_step=ws_step,
            wd_bin_overlap_radius=0.5,
        )

        # Get the underlying pandas data frame
        df_erb = er_out.df_result

        self.assertAlmostEqual(df_erb["baseline"].iloc[1], 0.807713, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[2], 0.884564, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[3], 0.921262, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[4], 0.942649, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[5], 0.959025, places=4)

        self.assertEqual(df_erb["count_baseline"].iloc[0], 1)
        self.assertEqual(df_erb["count_baseline"].iloc[1], 30)
        self.assertEqual(df_erb["count_baseline"].iloc[2], 44)
        self.assertEqual(df_erb["count_baseline"].iloc[3], 34)
        self.assertEqual(df_erb["count_baseline"].iloc[4], 38)
        self.assertEqual(df_erb["count_baseline"].iloc[5], 6)

    def test_flascdataframe_input(self):
        # Repeat the test above using a FlascDataFrame as input

        # Load data and FLORIS model
        fm, _ = load_floris()
        df = FlascDataFrame(load_data())
        df = dfm.set_wd_by_all_turbines(df)
        df_upstream = ftools.get_upstream_turbs_floris(fm)
        df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
        df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

        wd_step = 2.0
        ws_step = 1.0

        a_in = AnalysisInput([df], ["baseline"])

        er_out = erp.compute_energy_ratio(
            a_in,
            ["baseline"],
            test_turbines=[1],
            use_predefined_ref=True,
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_max=360.0,
            wd_min=0.0,
            wd_step=wd_step,
            ws_max=30.0,
            ws_min=0.0,
            ws_step=ws_step,
            wd_bin_overlap_radius=0.5,
        )

        # Get the underlying pandas data frame
        df_erb = er_out.df_result

        self.assertAlmostEqual(df_erb["baseline"].iloc[1], 0.807713, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[2], 0.884564, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[3], 0.921262, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[4], 0.942649, places=4)
        self.assertAlmostEqual(df_erb["baseline"].iloc[5], 0.959025, places=4)

        self.assertEqual(df_erb["count_baseline"].iloc[0], 1)
        self.assertEqual(df_erb["count_baseline"].iloc[1], 30)
        self.assertEqual(df_erb["count_baseline"].iloc[2], 44)
        self.assertEqual(df_erb["count_baseline"].iloc[3], 34)
        self.assertEqual(df_erb["count_baseline"].iloc[4], 38)
        self.assertEqual(df_erb["count_baseline"].iloc[5], 6)

    def test_energy_ratio_input(self):
        # Energy ratio input is deprecated but should still work

        # Load data and FLORIS model
        fm, _ = load_floris()
        df = FlascDataFrame(load_data())
        df = dfm.set_wd_by_all_turbines(df)
        df_upstream = ftools.get_upstream_turbs_floris(fm)
        df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
        df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])

        wd_step = 2.0
        ws_step = 1.0

        er_in = EnergyRatioInput([df], ["baseline"])

        er_out = erp.compute_energy_ratio(
            er_in,
            ["baseline"],
            test_turbines=[1],
            use_predefined_ref=True,
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_max=360.0,
            wd_min=0.0,
            wd_step=wd_step,
            ws_max=30.0,
            ws_min=0.0,
            ws_step=ws_step,
            wd_bin_overlap_radius=0.5,
        )

        # Get the underlying pandas data frame
        df_erb = er_out.df_result

        self.assertAlmostEqual(df_erb["baseline"].iloc[1], 0.807713, places=4)

    def test_row_reflection(self):
        from polars.testing import assert_frame_equal

        # Test adding reflected rows works as expected

        df = pl.DataFrame({"wd": [0.1, 0.5, 0.7], "ws": [6, 7, 8]})
        df_result_expected = pl.DataFrame({"wd": [0.1, 0.5, 0.7, 359.9], "ws": [6, 7, 8, 6]})
        edges = np.array([0, 2, 4])
        df_reflected = add_reflected_rows(df, edges, 0.25)
        assert_frame_equal(df_result_expected, df_reflected)

        df = pl.DataFrame({"wd": [359.1, 359.5, 359.9], "ws": [6, 7, 8]})
        df_result_expected = pl.DataFrame({"wd": [359.1, 359.5, 359.9, 0.1], "ws": [6, 7, 8, 8]})
        edges = np.array([358, 360])
        df_reflected = add_reflected_rows(df, edges, 0.25)
        assert_frame_equal(df_result_expected, df_reflected)

    def test_weight_by_min(self):
        # In the case we weight by min, there is 1 point in 7 m/s bin, 2 points in 8 m/s bin
        # so the test energy (001) should be (1 * 2) + (2 * 1) = 4
        # the ref energy (000) should be (1 * 1) + (2 * 1) = 3
        # And energy ratio = 4/3

        # Test the returned energy ratio assuming alternative weightings of the wind speed bins
        df_base = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [1.0, 1.0, 1.0, 1.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 7.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        er_out = erp.compute_energy_ratio(
            a_in,
            ref_turbines=[0],
            test_turbines=[1],
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_min=269.0,
            wd_step=2.0,
            ws_min=0.5,  # Make sure bin labels land on whole numbers
            weight_by="min",
        )

        self.assertAlmostEqual(er_out.df_result["wake_steering"].iloc[0], 4 / 3, places=4)

    def test_weight_by_sum(self):
        # In the case of weighting by sum there is 3 points in the 7 m /s bin and 5 points in the 8 m/s bin
        # so the test energy (001) should be (3 * 2) + (5 * 1) = 11
        # the ref energy (000) should be (3 * 1) + (5 * 1) = 8
        # And energy ratio = 11/8 (in df_wake_steering)

        df_base = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [1.0, 1.0, 1.0, 1.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 7.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        er_out = erp.compute_energy_ratio(
            a_in,
            ref_turbines=[0],
            test_turbines=[1],
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_min=269.0,
            wd_step=2.0,
            ws_min=0.5,  # Make sure bin labels land on whole numbers
            weight_by="sum",
        )

        self.assertAlmostEqual(er_out.df_result["wake_steering"].iloc[0], 11 / 8, places=4)

    def test_weight_by_sum_missing_bin_in_df(self):
        # This case tests that the energy ratio sum is properly weighted when one df (df_base in this case)
        # is missing a bin present in df_wake_steering

        # Computation should only include 7 m/s in this case since 8 m/s not included in df_base

        # In the case of weighting by sum there are 4 points in the 7 m /s bin and 2 points in the 8 m/s bin
        # But 8m/s should be excluded from df_base because not in df_base
        # so the test energy (001) should be (4 * 2) = 8
        # the ref energy (000) should be (4 * 1) = 4
        # And energy ratio = 2

        df_base = pd.DataFrame(
            {
                "wd": [270, 270.0],
                "ws": [7.0, 7.0],
                "pow_000": [1.0, 1.0],
                "pow_001": [1.0, 1.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 7.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        er_out = erp.compute_energy_ratio(
            a_in,
            ref_turbines=[0],
            test_turbines=[1],
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_min=269.0,
            wd_step=2.0,
            ws_min=0.5,  # Make sure bin labels land on whole numbers
            weight_by="sum",
        )

        self.assertAlmostEqual(er_out.df_result["wake_steering"].iloc[0], 2.0, places=4)

    def test_weight_by_external_frequency(self):
        df_base = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [1.0, 1.0, 1.0, 1.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 7.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        # In the final test, specify a bin frequency where 7 m/s is 90% and 8 m/s is 10%
        df_freq = pd.DataFrame({"wd": [270.0, 270.0], "ws": [7.0, 8.0], "freq_val": [0.9, 0.1]})

        er_out = erp.compute_energy_ratio(
            a_in,
            ref_turbines=[0],
            test_turbines=[1],
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_min=269.0,
            wd_step=2.0,
            ws_min=0.5,  # Make sure bin labels land on whole numbers
            df_freq=df_freq,
        )

        # In the case the weights come provided so can be used directly
        # so the test energy (001) should be (0.9 * 2) + (0.1 * 1) = 1.9
        # the ref energy (000) should be (0.9 * 1) + (.1 * 1) = 1
        # And energy ratio = 1.9 / 1
        self.assertAlmostEqual(er_out.df_result["wake_steering"].iloc[0], 1.9, places=4)

    def test_weight_by_external_frequency_with_extra_df_freq_bin(self):
        # Test that bins in df_freq which are not in data are ignored

        df_base = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [1.0, 1.0, 1.0, 1.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 7.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        # In the final test, specify uniform bin frequencies
        df_freq = pd.DataFrame(
            {
                "wd": [270.0, 270.0, 270.0, 270.0],
                "ws": [7.0, 8.0, 15.0, 20.0],
                "freq_val": [0.25, 0.25, 0.25, 0.25],
            }
        )

        er_out = erp.compute_energy_ratio(
            a_in,
            ref_turbines=[0],
            test_turbines=[1],
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_min=269.0,
            wd_step=2.0,
            ws_min=0.5,  # Make sure bin labels land on whole numbers
            df_freq=df_freq,
        )

        # In the case the weights come provided so can be used directly
        # so the test energy (001) should be (0.25 * 2) + (0.25 * 1) = .75
        # the ref energy (000) should be (0.25 * 1) + (.25 * 1) = .5
        # And energy ratio = .75 / .5 = 1.5
        self.assertAlmostEqual(er_out.df_result["wake_steering"].iloc[0], 1.5, places=4)

    def test_weight_by_external_frequency_with_missing_df_freq_bin(self):
        # Test the case where a bin in the data is not defined in df_freq
        # In this case the expected behavior is that bin missing from df_freq
        # get 0 weight and warning is printed

        df_base = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [1.0, 1.0, 1.0, 1.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 7.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        # Finally test the case where the weight of one of the bins is missing and defaults to 0
        # Here 6 and 7 m/s are specified but not 8, so the 8 m/s defaults to 0 weight
        df_freq = pd.DataFrame({"wd": [270.0], "ws": [7.0], "freq_val": [1.0]})

        er_out = erp.compute_energy_ratio(
            a_in,
            ref_turbines=[0],
            test_turbines=[1],
            use_predefined_wd=True,
            use_predefined_ws=True,
            wd_min=269.0,
            wd_step=2.0,
            ws_min=0.5,  # Make sure bin labels land on whole numbers
            df_freq=df_freq,
        )

        # Weight of 1.0 applied to 7 and 0 applied to 8
        # so the test energy (001) should be (1.0 * 2) + (0.0 * 1) = 2.
        # the ref energy (000) should be (1.0 * 1) + (0.0 * 1) = 1
        # And energy ratio = 2 / 1 -> 2
        self.assertAlmostEqual(er_out.df_result["wake_steering"].iloc[0], 2.0, places=4)

    def test_weight_by_external_frequency_with_all_missing_df_freq_bin(self):
        # Test the case where all bins in the data is not defined in df_freq
        # In this case an error should be raised

        df_base = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 8.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [1.0, 1.0, 1.0, 1.0],
            }
        )

        df_wake_steering = pd.DataFrame(
            {
                "wd": [
                    270,
                    270.0,
                    270.0,
                    270.0,
                ],
                "ws": [7.0, 7.0, 8.0, 8.0],
                "pow_000": [1.0, 1.0, 1.0, 1.0],
                "pow_001": [2.0, 2.0, 1.0, 1.0],
            }
        )

        a_in = AnalysisInput(
            [df_base, df_wake_steering], ["baseline", "wake_steering"], num_blocks=1
        )

        # Finally test the case where the weight of one of the bins is missing and defaults to 0
        # Here 6 and 7 m/s are specified but not 8, so the 8 m/s defaults to 0 weight
        df_freq = pd.DataFrame({"wd": [270.0], "ws": [10.0], "freq_val": [1.0]})

        with pytest.raises(RuntimeError):
            _ = erp.compute_energy_ratio(
                a_in,
                ref_turbines=[0],
                test_turbines=[1],
                use_predefined_wd=True,
                use_predefined_ws=True,
                wd_min=269.0,
                wd_step=2.0,
                ws_min=0.5,  # Make sure bin labels land on whole numbers
                df_freq=df_freq,
            )

    def test_null_behavior(self):
        # Test that in the default, an energy ratio is returned so long as any value is not null
        df = pd.DataFrame(
            {
                "wd_000": [
                    268.0,
                    270.0,
                    272.0,
                    272.0,
                ],
                "wd_001": [np.nan, 270.0, 272.0, 272.0],
                "ws": [8.0, 8.0, 8.0, 8.0],
                "pow_000": [100.0, 100.0, 100.0, 100.0],
                "pow_001": [100.0, np.nan, np.nan, np.nan],
                "pow_002": [100.0, 100.0, 200.0, np.nan],
            }
        )

        a_in_1 = AnalysisInput([df], ["baseline"], num_blocks=1)

        er_out_any = erp.compute_energy_ratio(
            a_in_1,
            ref_turbines=[0, 1],
            test_turbines=[2],
            wd_turbines=[0, 1],
            use_predefined_ws=True,
            wd_min=267.0,
            wd_step=2.0,
            ws_step=1.0,
            N=1,
        )

        df = pd.DataFrame(
            {
                "wd_000": [
                    268.0,
                    270.0,
                    np.nan,
                    272.0,
                ],
                "wd_001": [270.0, 270.0, 272.0, 272.0],
                "ws": [8.0, 8.0, 8.0, 8.0],
                "pow_000": [100.0, 100.0, 100.0, 100.0],
                "pow_001": [100.0, np.nan, np.nan, np.nan],
                "pow_002": [90.0, 100.0, 200.0, np.nan],
            }
        )

        a_in_2 = AnalysisInput([df], ["baseline"], num_blocks=1)

        er_out_all = erp.compute_energy_ratio(
            a_in_2,
            ref_turbines=[0, 1],
            test_turbines=[2],
            wd_turbines=[0, 1],
            use_predefined_ws=True,
            wd_min=267.0,
            wd_step=2.0,
            ws_step=1.0,
            N=1,
            remove_all_nulls=True,
        )

        with pytest.raises(RuntimeError):
            # Expected to fail because no bins remain after null filtering
            erp.compute_energy_ratio(
                a_in_1,
                ref_turbines=[0, 1],
                test_turbines=[2],
                wd_turbines=[0, 1],
                use_predefined_ws=True,
                wd_min=267.0,
                wd_step=2.0,
                ws_step=1.0,
                N=1,
                remove_all_nulls=True,
            )

        # Check outputs match expectations
        self.assertAlmostEqual(er_out_any.df_result["baseline"].iloc[0], 1.0, places=4)
        self.assertAlmostEqual(er_out_any.df_result["baseline"].iloc[1], 1.0, places=4)
        self.assertAlmostEqual(er_out_any.df_result["baseline"].iloc[2], 2.0, places=4)

        self.assertEqual(er_out_any.df_result["count_baseline"].iloc[0], 1)
        self.assertEqual(er_out_any.df_result["count_baseline"].iloc[1], 1)
        self.assertEqual(er_out_any.df_result["count_baseline"].iloc[2], 1)

        self.assertAlmostEqual(er_out_all.df_result["baseline"].iloc[0], 0.9, places=4)

        self.assertEqual(er_out_all.df_result["count_baseline"].iloc[0], 1)
