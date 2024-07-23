# Getting started

The easiest way to get started is to install FLASC and
then follow the examples. The correct order is:

## Install FLASC
Install the repository following the instructions `here <installation.html>`_.

## FLASC examples
You can generate a demo dataset by following the examples in
``examples_smarteole/``. The notebook ``02_download_and_format_dataset.ipynb``
downloads data from a wake steering experiment conducted in 2019. We encourage
users to step through the notebooks in ``examples_smarteole/`` in order to
develop an understanding of FLASC's capabilities using a dataset from a real
field experiment.

Additional useful examples can be found in ``examples_artificial_data/``, where
we intentionally introduce "challenges" for the FLASC tools to solve using
artificially-generated data. This provides a good way for users to get to know
the FLASC tools in more depth. Again, we recommend stepping through the
examples in the subdirectories in their numerical order.

Roughly speaking, the examples in both ``examples_smarteole/`` and
``examples_artificial_data`` demonstrate the FLASC modules in the order:
- `flasc.data_processing`
- `flasc.analysis`
- `flasc.model_fitting`

and use `flasc.utilities` throughout.



## 

FLASC is a tool for processing and running analysis on wind farm SCADA. 
If you have SCADA data from a wind farm, FLASC could be the tool for you! FLASC is written in python and relies heavily on [`pandas`](https://pandas.pydata.org/). We assume a working knowledge of both.

The general steps you'll probably want to follow are
1. Get your SCADA data into a `pandas` `DataFrame`. FLASC assumes that the dataframe is in "wide" format (see notes below), and has at least the following columns:
- `pow_XXX` Turbine power, specified in kW (TODO: check unit)
- `ws_XXX` Turbine-measured wind speed, specified in m/s
- `wd_XXX` Turbine-measured wind direction, specified in deg (you may not have this channel, in which case, we suggest you use the yaw position as a proxy)
- `yaw_XXX` Turbine yaw position, specified in deg 
Additionally, you may have other useful channels, such as the operating status, low-speed or high-speed shaft speed, blade pitch position, etc. You may also have data from other measurement devices such as lidars or met masts.
2. Use `FLASC`'s `data_processing` package to filter your data, removing data such as: 
- off power-curve measurements
- stuck sensors
- bad status flags
- unusual blade pitch or rotor speed signals
3. Apply corrections---turbine data often contains a "northing" bias, i.e., a miscalibrated yaw encoder, which `FLASC` has tools to correct this 
issue.
4. Fit a [`FLORIS`](https://github.com/NREL/floris) model to your data using `FLASC`'s `model_fitting` package. `FLASC` has tools to:
- fit the cosine loss exponent for yawing (useful when analyzing data from 
a wake steering campaign or designing a wake steering controller)
- fit wake model parameters, particularly those of the 
[Empirical Gaussian wake model](https://nrel.github.io/floris/empirical_gauss_model.html), so that you can better understand wake interactions in your farm and compare model predictions to realized wake losses.
5. Run energy ratio and total uplift analysis with tools from `FLASC`'s `analysis` package. These allow you to quantify the difference between cases seen in your data (for example, `wake_steering_on` and `wake_steering_off`, or `daytime` and `nighttime`) in terms of:
- The energy produced at each turbine as a function of wind direction, particularly when wake interactions are present
- The total power difference between cases


## Getting started
---------------

The easiest way to get started is to install FLASC and
then follow the examples. The correct order is:

Install FLASC
=============
Install the repository following the instructions `here <installation.html>`_.

Generating an artificial dataset
================================
You can generate a demo dataset using the script at
``examples_artificial_data/demo_dataset/``. The script ``generate_demo_dataset.py`` downloads
historical data from a meteorological measurement tower at the U.S. National
Wind Technology Center (NWTC), part of the National Renewable Energy
Laboratory (NREL). This data is made readily available to the public on its
corresponding `website <https://midcdmz.nrel.gov/>`_. This script downloads
historical data for the entire year of 2019 at approximately 60 second
intervals. We then assume the wind direction measured by this met tower
is equal to the wind direction at every wind turbine in the wind farm.
Further, we derive each turbine's power production, wind speed and turbulence
intensity using the floris wind farm model.

We then introduce realistic disturbances on these measurements. Firstly, we
add randomized noise. We also add curtailment periods to certain turbines,
in which the power production is saturated to a below-rated value for a
period of time. We also add other realistic noise on top of the turbine
wind-speed power curves. We then mark several weeks of the year for
a handful of turbines are completely faulty, mimicking turbine downtime,
for example for maintenance. We add northing errors to all of the turbines,
which is common in field data. We also mimic sensor-stuck type of faults,
in which certain sensors report the exact same measurement for unrealistically
long periods of time (e.g., a vane signal reporting 13.44 deg for 3 minutes
consecutively). Finally, we also add a measurement tower with wind direction,
wind speed and turbulence intensity measurements, and time-shift this data
by 2 hours. This represents the realistic situation in which external
measurement equipment and the turbine's internal logger follow a different
clock or timezone.

Raw data processing
===================
Once the artificial dataset has been generated, it should be filtered and
postprocessed using the steps described in
`Raw data processing <data_processing.html>`_. This will filter the SCADA data
for measurement outliers and deal with northing calibration.


Data-based analysis
===================
Once the artificial dataset has been postprocessed, one can start performing
analyses. See `Data analysis <data_analysis.html>`_ for examples on how to
derive useful information from the data, such as for model tuning and model
validation.

.. seealso:: `Return to table of contents <index.html>`_
