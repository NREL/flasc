import os
import matplotlib.pyplot as plt
import numpy as np

from flasc import floris_tools as fsatools
from flasc import visualization as fsaviz

from floris import tools as wfct

# Demonstrate the turbine dependency functions in floris_tools
# Note a turbine is "dependent" on another if it is affected 
# by the wake of the other turbine for a given wind direction.

# A given turbine's dependent turbines are those that depend on it,
# and a turbine's impacting turbines are those turbines that
# it itself depends on.

    
# Set up FLORIS interface
print('Initializing the FLORIS object for our demo wind farm')
file_path = os.path.dirname(os.path.abspath(__file__))
fi_path = os.path.join(file_path, '../demo_dataset/demo_floris_input.yaml')
fi = wfct.floris_interface.FlorisInterface(fi_path)

# Plot the layout of the farm for reference
fsaviz.plot_layout_only(fi)

# Get the dependencies of turbine 2
check_directions = np.arange(0, 360., 2.)
depend_on_2 = fsatools.get_dependent_turbines_by_wd(fi, 2, check_directions)

print("Turbines that depend on T002 at 226 degrees:", 
      depend_on_2[round(226/2)]
     )

# Can also return all influences as a matrix for other use (not ordered)
depend_on_2, influence_magnitudes = fsatools.get_dependent_turbines_by_wd(
    fi, 2, check_directions, return_influence_magnitudes=True)
print("\nArray of all influences of T002 has shape (num_wds x num_turbs): ", 
      influence_magnitudes.shape)
print("Influence of T002 on T006 at 226 degrees: {0:.4f}".format( 
      influence_magnitudes[round(226/2), 6]))

df_dependencies = fsatools.get_all_dependent_turbines(fi, check_directions)
print("\nAll turbine dependencies using default threshold "+\
      "(first 5 wind directions printed):")
print(df_dependencies.head())

df_dependencies = fsatools.get_all_dependent_turbines(fi, check_directions, 
    limit_number=2)
print("\nTwo most significant turbine dependencies using default threshold "+\
      "(first 5 wind directions printed):")
print(df_dependencies.head())

df_dependencies = fsatools.get_all_dependent_turbines(fi, check_directions, 
    change_threshold=0.01)
print("\nAll turbine dependencies using higher threshold "+\
      "(first 5 wind directions printed):")
print(df_dependencies.head())

print("\nAll upstream turbine impacts using default threshold "+\
      "(first 5 wind directions printed):")
df_impacting = fsatools.get_all_impacting_turbines(fi, check_directions)
print(df_impacting.head())
# Inclusion of T005 here as an impact on T000 is surprising; try increasing
# the threshold or reducing the limit_number (see next).

print("\nMost significant upstream turbine impact using default threshold "+\
      "(first 5 wind directions printed):")
df_impacting = fsatools.get_all_impacting_turbines(fi, check_directions,
    limit_number=1)
print(df_impacting.head())

print("\nAll upstream turbine impacts using higher threshold "+\
      "(first 5 wind directions printed):")
df_impacting = fsatools.get_all_impacting_turbines(fi, check_directions,
    change_threshold=0.01)
print(df_impacting.head())

# Note that there is no individual turbine version for the "impacting" 
# function; instead, compute all impacting turbines and extract desired 
# turbine from the output dataframe.

# (compute using defaults again, for example)
df_impacting = fsatools.get_all_impacting_turbines(fi, check_directions)
print("\nTurbines that T006 depends on at 226 degrees:", 
      df_impacting.loc[226, 6]
     )


plt.show()
