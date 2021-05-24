import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris_scada_analysis import yaw_pow_fitting as ywpf


N = 10000
ws = np.ones(N) * 8.
cp = 0.45

vane = 5. * np.random.randn(N)
pow = 0.5 * 1.225 * (.25 * np.pi * 120.**2) * ws**3 * cp * np.cos(vane * np.pi / 180.)**2.0

vane_ref = 5. * np.random.randn(N)
pow_ref = 0.5 * 1.225 * (.25 * np.pi * 120.**2) * ws**3 * cp * np.cos(vane_ref * np.pi / 180.)**2.0

df_upstream = pd.DataFrame({
    'wd_min': [0.],
    'wd_max': [360.],
    'turbines': [[0, 1]]
    })

df = pd.DataFrame({
    'wd': np.ones(N),
    'ws': ws,
    'vane_000': vane,
    'vane_001': vane_ref,
    'pow_000': pow,
    'pow_001': pow_ref,
})

# Initialize yaw-power curve filtering
yaw_pow_filtering = ywpf.yaw_pow_fitting(
    df, df_upstream, turbine_list=[0])
yaw_pow_filtering.calculate_curves()
yaw_pow_filtering.estimate_cos_pp_fit()
yaw_pow_filtering.plot()
plt.show()