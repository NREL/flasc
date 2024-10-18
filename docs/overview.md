# Overview

FLASC provides a rich suite of analysis tools for SCADA data filtering &
analysis, wind farm model validation, field experiment design, and field
experiment monitoring. The repository is centrally built around NREL's
in-house [FLORIS](https://github.com/nrel/floris) wake modeling utility.
FLASC also largely relies on the "energy ratio" to quantify wake
losses in synthetic and historical data, perform turbine northing
calibrations, and for model parameter estimation.

# FLASC package

FLASC consists of multiple modules, including:


## flasc.data_processing

This module contains functions that supports importing and processing raw
SCADA data files. Data is saved in feather
format for optimal balance of storage size and load and write speed.

Functions include filtering data by wind direction, wind speed an/or TI,
deriving the ambient conditions from the upstream turbines, all the while
dealing with angle wrapping for angular variables. Outliers
can be detected and removed at the turbine level.
Filtering methods include sensor-stuck type of
fault detection and analysis of the turbine wind speed-power curve.

Also included are functions to downsample, upsample and calculate
moving averages of a data frame with SCADA and/or FLORIS data. These functions
allow the user to specify which columns contain angular variables, and
consequently 360 deg wrapping is taken care of. It also allows the user
to calculate the min, max, std and median for downsampled data frames. It
leverages efficient functions inherent in pandas and polars to maximize
performance.

Finally, functions are provided to detect northing bias (caused by
miscalibrated yaw encoders) in turbine data.


## flasc.analysis

This module contains classes to calculate and visualize the energy ratio as
defined by Fleming et al. (2019). The energy ratio is a very useful quantity
in SCADA data analysis and related model validation. It represents the amount
of energy produced by a turbine relative to what that turbine would have
produced if no wakes were present.  See [energy ratio](energy_ratio) for more
details. Also included are methods for calculating the total power uplift in a
comparative field experiment.

## flasc.utilities

This module contains utilities that support the other modules within FLASC.
These utilities help to interface with FLORIS and calculate a large set of
floris simulations for different atmospheric conditions, yaw misalignments
and/or model parameters. It also includes two functions to precalculate and
respectively interpolate from a large set of model solutions to speed up
further postprocessing.

Also included are functions to estimate the timeshift between
two sources of data, for example, to sychronize measurements from a met mast
with measurements from SCADA data. The module also includes a function to
estimate the offset between two timeseries of wind direction measurements.
This is useful to determine the northing bias of a turbine if you know the
correct calibration of at least one other wind turbine. Finally, this module
also contains a function to estimate the atmospheric turbulence intensity
based on the power measurements of the turbines inside a wind farm.

Additionally, visualization tools can be found in `flasc.visualization` and `flasc.yaw_optimizer_visualization.`

# Literature

See {cite:p}`Doekemeijer2022a` and {cite:p}`Bay2022a`for practical
examples of how the flasc repository is used or processing and analyses of
historical SCADA data of three offshore wind farms.

  ```{bibliography}
  ```

# Citation


If FLASC played a role in your research, please cite it. This software can be
cited as:

   FLASC. Version 2.0.1 (2024). Available at https://github.com/NREL/flasc.

For LaTeX users:


    @misc{flasc2024,
      author = {NREL},
      title = {FLASC. Version 2.0.1},
      year = {2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      url = {https://github.com/NREL/flasc},
    }


# Questions

For technical questions regarding FLASC usage, please post your questions to
[GitHub Discussions](https://github.com/NREL/flasc/discussions) on the
FLASC repository. Alternatively, email the NREL FLASC team at
`paul.fleming@nrel.gov <mailto:paul.fleming@nrel.gov>`_ or
`michael.sinner@nrel.gov <mailto:michael.sinner@nrel.gov>`_.
