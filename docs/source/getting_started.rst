Getting started
---------------

The easiest way to get started is to install FLASC and
then follow the examples. The correct order is:

Install FLASC
=============
Install the repository following the instructions `here <installation.html>`_.

Generating an artificial dataset
================================
You can generate a demo dataset using the script at
``examples/demo_dataset/``. The script ``generate_demo_dataset.py`` downloads
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
