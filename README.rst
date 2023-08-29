==========
FLORIS-Based Analysis for SCADA Data (FLASC)
==========

**Note:** Further documentation is available at **http://flasc.readthedocs.io/.**

Description
-----------

FLASC provides a **rich suite of analysis tools** for SCADA data filtering, analysis, 
wind farm model validation, field experiment design, and field experiment monitoring. 

The repository is centrally built around NRELs in-house ``FLORIS`` wind farm model, available at
**https://github.com/nrel/floris**. FLASC also largely relies on the ``energy ratio``, among others, 
to quantify wake losses in synthetic and historical data, to perform turbine northing calibrations, 
and model parameter estimation.

For technical questions or concerns, please email paul.fleming@nrel.gov.

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

FLASC consists of several modules, including:

* **flasc.dataframe_operations**: Contains functions to easily manipulate Pandas dataframes. This includes functions for filtering data by wind direction, wind speed and/or TI, and deriving the ambient conditions from the upstream turbines, all while handling angle wrapping for angular variables.

* **flasc.energy_ratio**: Contains classes to calculate and visualize the *energy ratio* as defined by Fleming et al. (2019). The ``energy ratio`` is an important quantity in SCADA data analysis and related model validation. It represents the amount of energy produced by a turbine *relative* to what that turbine would have produced if *no* wakes were present. Various classes are included in this model, such as classes for calculating and plotting the energy ratio for single or multiple datasets, and a class that calculates the wind direction bias for every turbine by maximizing the energy ratio fit between FLORIS and SCADA data. There are also visualization functions for energy ratio plots and automated generation of detailed excel spreadsheets to determine where and which turbines performed differently than expected. These functions can be used both for model validation and processing field campaign data (e.g. baseline vs. controlled operation).

* **flasc.floris_tools**: Contains functions that leverage the FLORIS model *directly*. This includes functions to calculate a large set of FLORIS simulations (with MPI, optionally) for different atmospheric conditions, yaw misalignments, and/or model parameters. There are also functions to pre-calculate and respectively interpolate from a large dataset of model solutions to improve the efficiency of further post-processing.

* **flasc.model_estimation**: Contains classes pertaining to the estimation of parameters in the FLORIS wind farm model. The class ``floris_sensitivity_analysis``, performs Sobol parameter sensitivity studies to determine which parameters are most sensitive in certain situations (e.g. atmospheric conditions, turbine settings, wind farm layouts).

* **flasc.optimization**: Contains functions that estimate the time shift between two sources of data, for example, to synchronize measurements from a met mast with measurements from SCADA data. Includes a function to estimate the offset between two timeseries of wind direction measurements. This is useful to determine the northing bias of a turbine assuming the user has knowledge about the correct calibration of at *least* one turbine. Also includes a function to estimate the atmospheric turbulence intensity based on the power measurements of the turbines inside a wind farm.

* **flasc.raw_data_handling**: Contains functions that support importing and processing raw SCADA data files. The class ``sql_database_manager`` up/downloads data between the local file system and a remote SQL database. This class also contains a GUI to visualize data existent in the remote repository. This repository also includes data handling for very large datasets. Data is saved using feather format for an *optimal* balance of storage size and read/write speed. Additionally, large dataframes can be *split* up into multiple dataframes and feather files.

* **flasc.time_operations**: Contains functions for downsampling, upsampling and calculating moving averages of a dataframe containing SCADA and/or FLORIS data. These functions allow the user to specify which columns contain angular variables, and consequently handle 360 deg wrapping. Additionally, these functions can calculate the min, max, std and median for downsampled dataframes. This module leverages efficient functions *native* to Pandas to maximize performance.

* **flasc.turbine_analysis**: Contains functions that enable SCADA data analysis on the *turbine* level. Outliers can be detected and removed. Filtering-specific functions include sensor-stuck type fault detection and analysis of the turbine wind speed-power curve.

* **flasc.model_tuning**: Contains a class for tuning FLORIS model parameters to yield results that *align* with SCADA data. This class offers a suite of functions for tuning, visualizations that compare FLORIS to SCADA and outputting the tuned parameters to a YAML file for later use. 

License
------------

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   **http://www.apache.org/licenses/LICENSE-2.0**

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
