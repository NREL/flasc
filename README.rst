=============================================
FLORIS-based Analysis for SCADA data (FLASC)
=============================================

**Further documentation is available at http://flasc.readthedocs.io/.**

FLASC provides a rich suite of analysis tools for SCADA data
filtering & analysis, wind farm model validation, field experiment design,
and field experiment monitoring. The repository is centrally built around
NRELs in-house `floris` wind farm model, available at
https://github.com/nrel/floris. FLASC also largely relies on the `energy ratio`
to, among others, quantify wake losses in synthetic and historical data, to
perform turbine northing calibrations, and for model parameter estimation.

For technical questions, please send an email to paul.fleming@nrel.gov.

.. image:: https://readthedocs.org/projects/flasc/badge/?version=main
   :target: http://flasc.readthedocs.io/
   :alt: Documentation status

.. image:: https://github.com/NREL/flasc/actions/workflows/continuous-integration-workflow.yaml/badge.svg?branch=main
   :target: https://github.com/NREL/flasc/actions
   :alt: Automated tests success

.. image:: https://codecov.io/gh/nrel/flasc/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/nrel/flasc/
   :alt: Code coverage

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code formatting style

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: Apache 2.0 License

Installation
------------
We recommend installing this repository in a separate virtual environment.
After creating a new virtual environment, clone this repository to your local
system and install it locally using ``pip``. The command for this is ``pip install -e flasc``.
    
Features
--------
FLASC consists of multiple modules, including:

* *flasc.dataframe_operations*: This module includes functionality to easily manipulate pandas data frames. Functions include filtering data by wind direction, wind speed an/or TI, deriving the ambient conditions from the upstream turbines, all the while dealing with angle wrapping for angular variables.
* *flasc.energy_ratio*: this module contains classes to calculate and visualize the energy ratio as defined by Fleming et al. (2019). The energy ratio is a very useful quantity in SCADA data analysis and related model	validation. It represents the amount of energy produced by a turbine relative to what that turbine would have produced if no wakes were present. Various classes are included in this model, from classes used to calculate and plot the energy ratio for a single dataset, a class for multiple datasets, and a class that calculates the wind direction bias for every turbine by maximizing the energy ratio fit between FLORIS and SCADA data. Various visualization methods are included such as energy ratio plots and automated generation of detailed excel spreadsheets to determine where and which turbines performed differently than expected. These methods	can be used both for model validation and for processing field campaign data, e.g. baseline vs optimized operation.
* *flasc.floris_tools*: this module contains functions that leverage the floris model directly. This includes functions to calculate a large set of floris simulations (with MPI, optionally) for different atmospheric conditions, yaw misalignments and/or model parameters. It also includes two functions	to precalculate and respectively interpolate from a large set of model solutions to speed up further postprocessing.
* *flasc.model_estimation*: This is a module related to the estimation of parameters in the floris wind farm model. One class herein, called `floris_sensitivity_analysis`, performs Sobol parameter sensitivity	studies to determine which parameters are most sensitive in various situations (atmospheric conditions, turbine settings, wind farm layouts).
* *flasc.optimization*: The optimization module includes functions to estimate the timeshift between two sources of data, for example, to sychronize measurements from a met mast with measurements from SCADA data. The module also includes a function to estimate the offset between two timeseries of wind direction measurements. This is useful to determine the northing bias of a turbine if you know the correct calibration of at least one other wind turbine. Finally, this module also contains a function to estimate the atmospheric turbulence intensity based on the power measurements of the turbines inside a wind farm.
* *flasc.raw_data_handling*: this module contains functions that supports importing and processing raw SCADA data files. Specifically, it provides a class called "sql_database_manager" which can be used to up- and download data between your local system and a remote SQL database. This class also  contains a GUI to visualize data existent in the remote repository. This repository also includes data handling for very large datasets. Data is saved in feather format for optimal balance of storage size and load/write speed. Additionally, can split one large dataframe into multiple dataframes and feather files.
* *flasc.time_operations*: This module allows the user to easily downsample, upsample and calculate moving averages of a data frame with SCADA and/or FLORIS data. These functions allow the user to specify which columns contain angular variables, and consequently 360 deg wrapping is taken care	of. It also allows the user to calculate the min, max, std and median for downsampled data frames. It leverages efficient functions inherent in	pandas to maximize performance.
* *flasc.turbine_analysis*: this module allows the user to analyze SCADA data on a turbine level. Outliers can be detected and removed. Filtering methods include sensor-stuck type of fault detection and analysis of the turbine wind speed-power curve.

License
------------

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
