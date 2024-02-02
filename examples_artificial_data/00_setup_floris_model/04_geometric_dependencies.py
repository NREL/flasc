import matplotlib.pyplot as plt
import numpy as np
from floris.tools import FlorisInterface

from flasc.utilities.floris_tools import get_all_impacting_turbines_geometrical

# Demonstrate the get_all_impacting_turbines_geometrical
# function in floris_tools

# Load a large FLORIS object
fi = FlorisInterface("../floris_input_artificial/gch.yaml")
D = 126.0
X, Y = np.meshgrid(7.0 * D * np.arange(20), 5.0 * D * np.arange(20))
fi.reinitialize(layout_x=X.flatten(), layout_y=Y.flatten())

# Specify which turbines are of interest
turbine_weights = np.zeros(len(X.flatten()), dtype=float)
turbine_weights[np.hstack([a + range(10) for a in np.arange(50, 231, 20)])] = 1.0

# Get all impacting turbines for each wind direction using simple geometry rules
df_impacting = get_all_impacting_turbines_geometrical(
    fi=fi, turbine_weights=turbine_weights, wd_array=np.arange(0.0, 360.0, 30.0)
)

# Produce plots showcasing which turbines are estimated to be impacting
for ii in range(df_impacting.shape[0]):
    wd = df_impacting.loc[ii, "wd"]

    fig, ax = plt.subplots()
    ax.plot(fi.layout_x, fi.layout_y, "o", color="lightgray", label="All turbines")

    ids = df_impacting.loc[ii, "impacting_turbines"]
    no_turbines_total = len(fi.layout_x)
    no_turbines_reduced = len(ids)
    ax.plot(fi.layout_x[ids], fi.layout_y[ids], "o", color="black", label="Impacting turbines")

    ids = np.where(turbine_weights > 0.001)[0]
    ax.plot(fi.layout_x[ids], fi.layout_y[ids], "o", color="red", label="Turbines of interest")

    ax.set_xlabel("X location (m)")
    ax.set_ylabel("Y location (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    percentage = 100.0 * no_turbines_reduced / no_turbines_total
    ax.set_title(
        f"Wind direction: {wd:.1f} deg. Turbines modelled: "
        f"{no_turbines_reduced:d}/{no_turbines_total} ({percentage:.1f}%)."
    )

    # Make a statement on number of wake-steered turbines vs. total farm size
    print(
        f"wd={wd:.1f} deg. Reduced from {no_turbines_total:d} "
        f"to {no_turbines_reduced} ({percentage:.1f}%)."
    )

plt.show()
