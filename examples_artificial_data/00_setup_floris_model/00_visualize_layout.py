import matplotlib.pyplot as plt
import numpy as np
import floris.layout_visualization as layoutviz

from flasc.utilities.utilities_examples import load_floris_artificial as load_floris

# Example demonstrates some methods for visualizing the layout of the farm
# represented within the FLORIS interface


if __name__ == "__main__":
    # Set up FLORIS interface
    print("Initializing the FLORIS object for our demo wind farm")
    fm, _ = load_floris()

    # Defines alternative names for each turbine with 1-index
    turbine_names = ["Turbine-%d" % (t + 1) for t in range(len(fm.layout_x))]

    # Plot the basic farm layout
    ax = layoutviz.plot_turbine_points(fm, plotting_dict={"color": "g"})
    layoutviz.plot_turbine_labels(fm, ax=ax, turbine_names=turbine_names)
    ax.grid()
    ax.set_xlabel("x coordinate [m]")
    ax.set_ylabel("y coordinate [m]")
    
    # Plot using the default names and show the wake directions
    turbines_to_plot = range(2, len(fm.layout_x))
    ax = layoutviz.plot_turbine_points(fm, turbine_indices=turbines_to_plot)
    layoutviz.plot_turbine_labels(fm, ax=ax, turbine_indices=turbines_to_plot, turbine_names=turbine_names)
    layoutviz.plot_waking_directions(
        fm, 
        ax=ax,
        limit_num=3,
        wake_plotting_dict={"color": "r"},
        turbine_indices=turbines_to_plot,
    )
    ax.grid()
    ax.set_xlabel("x coordinate [m]")
    ax.set_ylabel("y coordinate [m]")

    # Demonstrate shading of an arbitrary region
    points_for_demo = np.array([[600, 0], [1400, 0], [1200, 1000]])
    ax = layoutviz.plot_turbine_points(fm)
    layoutviz.plot_turbine_labels(fm, ax=ax, turbine_names=turbine_names)
    layoutviz.shade_region(
        points_for_demo,
        show_points=True,
        plotting_dict_region={"color": "blue", "label": "Example region"},
        plotting_dict_points={"color": "blue", "marker": "+", "s": 50},
        ax=ax,
    )
    ax.grid()
    ax.set_xlabel("x coordinate [m]")
    ax.set_ylabel("y coordinate [m]")

    # Turbine data
    fig, ax = plt.subplots(2,1)
    ax[0].plot(
        fm.core.farm.turbine_map[0].power_thrust_table["wind_speed"],
        fm.core.farm.turbine_map[0].power_thrust_table["power"],
    )
    ax[1].plot(
        fm.core.farm.turbine_map[0].power_thrust_table["wind_speed"],
        fm.core.farm.turbine_map[0].power_thrust_table["thrust_coefficient"],
    )
    ax[1].set_xlabel("Wind Speed [m/s]")
    ax[1].set_ylabel("Thrust coefficient [-]")
    ax[0].set_ylabel("Power [kW]")
    ax[0].grid()
    ax[1].grid()

    plt.show()
