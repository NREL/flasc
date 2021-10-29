import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from floris import tools as wfct
from floris.utilities import wrap_360
from floris_scada_analysis.energy_ratio import energy_ratio_suite
from floris_scada_analysis.dataframe_operations import \
    dataframe_manipulations as dfm
from floris_scada_analysis import floris_tools as fsatools
from floris_scada_analysis.energy_ratio import energy_ratio_visualization as erv

def load_data():
    # Load dataframe with scada data
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ftr_path = os.path.join(root_dir, '..', 'demo_dataset',
                            'demo_dataset_60s.ftr')
    if not os.path.exists(ftr_path):
        raise FileNotFoundError('Please run ./examples/demo_dataset/' +
                                'generate_demo_dataset.py before try' +
                                'ing any of the other examples.')
    df = pd.read_feather(ftr_path)
    return df


def load_floris():
    # Initialize the FLORIS interface fi
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, "../demo_dataset/demo_floris_input.json")
    fi = wfct.floris_interface.FlorisInterface(fi_path)
    return fi


# Load data and FLORIS
df = load_data()
fi = load_floris()

# Visualize layout
fi.vis_layout()

# We first need to define a wd against which we plot the energy ratios
# In this example, we set the wind direction to be equal to the mean
# wind direction between all turbines
df = dfm.set_wd_by_all_turbines(df)


# # We reduce the dataframe to only data where the wind direction
# # is between 0 and 90 degrees.
# df = dfm.filter_df_by_wd(df=df, wd_range=[0., 90.])
# df = df.reset_index(drop=True)

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
df2['wd'] = wrap_360(df2['wd'] + np.random.normal(0,1.,df2.wd.values.shape))

# Also make a gap between 8 and 10 m/s
df2 = df2[(df2.ws < 8) | (df2.ws > 10)]

# df2['ws'] = df2['ws'] + np.random.normal(0,0.5,df2.ws.values.shape)

# Initialize the energy ratio suite object and add each dataframe
# separately. We will import the original data and the manipulated
# dataset.
fsc = energy_ratio_suite.energy_ratio_suite()
fsc.add_df(df, 'Baseline')
fsc.add_df(df2, 'Random WD Perturbation')

# Print the dataframes to see if everything is imported properly
fsc.print_dfs()

# Make the energy table
# Choose the ws and wd bins
wd_bins = np.arange(40,50,2)
ws_bins = np.arange(5,12,1)

# Select test turbines
test_turbines = [1,4]

fsc.get_energy_tables(
        test_turbines,
        wd_bins,
        ws_bins,
        excel_filename = 'energy_table.xlsx',
        fi = fi,
        verbose=False
    )


# plt.show()