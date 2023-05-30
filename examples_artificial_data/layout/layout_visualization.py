# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os

import matplotlib.pyplot as plt
import numpy as np

from flasc.visualization import (
    plot_floris_layout, 
    plot_layout_only, 
    plot_layout_with_waking_directions,
    shade_region
)

from floris import tools as wfct

# Example demonstrates some methods for visualizing the layout of the farm
# represented within the FLORIS interface


if __name__ == "__main__":

    # Set up FLORIS interface
    print('Initializing the FLORIS object for our demo wind farm')
    file_path = os.path.dirname(os.path.abspath(__file__))
    fi_path = os.path.join(file_path, '../demo_dataset/demo_floris_input.yaml')
    fi = wfct.floris_interface.FlorisInterface(fi_path)

    # Defines alternative names for each turbine with 1-index
    turbine_names = ['Turbine-%d' % (t + 1) for t in range(len(fi.layout_x))]


    # Plot using default 0-indexed labels (includes power/thrust curve)
    plot_floris_layout(fi, plot_terrain=False) 

    # Plot using default given 1-indexed labels (includes power/thrust curve)
    plot_floris_layout(fi, plot_terrain=False, turbine_names=turbine_names) 

    # Plot only the layout with default options
    plot_layout_only(fi)

    # Plot only the layout with custom options
    plot_layout_only(fi,
        {
            'turbine_names':turbine_names,
            'color':'g'
        }
    )

    # Plot layout with wake directions and inter-turbine distances labeled
    plot_layout_with_waking_directions(fi)

    # Plot layout with wake directions and inter-turbine distances labeled
    # (using custom options)
    plot_layout_with_waking_directions(fi,
        limit_num = 3, # limit to 3 lines per turbine
        layout_plotting_dict = {'turbine_names':turbine_names, 
                                'turbine_indices':range(2,len(fi.layout_x))},
        wake_plotting_dict={'color':'r'}
    )

    # Demonstrate shading of an arbitrary region
    points_for_demo = np.array([[600, 0], [1400, 0], [1200, 1000]])
    ax = plot_layout_only(fi)
    shade_region(
        points_for_demo, 
        show_points=True,
        plotting_dict_region={"color":"blue"},
        plotting_dict_points={"color":"blue", "marker":"+", "s":50},
        ax=ax
    )

    plt.show()
