# Calculating the change in energy production

To investigate the change in energy production from a field test of a technology such as wind farm control, the FLASC repository was originally built around [energy ratio analysis](energy_ratio) with a focus on observing patterns in changes in energy production and visually comparing those with those changes expected from wake models.

FLASC now however includes three methods for quantifying the change in energy production.

## Total Uplift Power Ratio

`total_uplift_power_ratio` uses a similar input as the [energy ratio](energy_ratio) methods but returns a single value representing the total uplift, rather then binned by wind direction.  The method is named power_ratio because the change in energy production is computed the mean per-wind-condition-bin power ratio is computed.  The change in ratio per bin is then combined with the mean base power and frequency to estimate change in energy production.

Currently the main example usage of the total uplift function is at the end of [smarteole example 06](https://github.com/NREL/flasc/blob/main/examples_smarteole/06_wake_steering_energy_ratio_analysis.ipynb).  Documentation of the function itself is available in the [API documentation](https://nrel.github.io/flasc/_autosummary/flasc.analysis.total_uplift_power_ratio.compute_total_uplift.html#flasc.analysis.total_uplift_power_ratio.compute_total_uplift).  Uncertainty of the results can be computed via bootstrapping.

The method was developed by Eric Simley and implemented by Paul Fleming and Misha Sinner of NREL.  

## Wind-Up

FLASC further includes methods for calculating change in energy production using the [wind-up](https://github.com/resgroup/wind-up) module.   [wind-up](https://github.com/resgroup/wind-up) is a tool to assess yield uplift of wind turbines developed by Alex Clerc of RES and available open-source on GitHub.  Using translation methods in the [FlascDataFrame](flasc_data_format), the methods and analysis of wind-up can be invoked from FLASC.

[smarteole example 09](https://github.com/NREL/flasc/blob/main/examples_smarteole/09_wind-up_wake_steering_uplift_analysis.ipynb), calculates the change in energy production (as in [smarteole example 06](https://github.com/NREL/flasc/blob/main/examples_smarteole/06_wake_steering_energy_ratio_analysis.ipynb)) using wind-up.

## Expected Power Analysis

The final included methodology for calculating change in energy production is the module `expected_power_analysis`.  This module implements the calculations of change in energy production described in [AWC validation methodology](https://publications.tno.nl/publication/34637216/LWOb3s/TNO-2020-R11300.pdf), by Stoyan Kanev of TNO.  The method was implemented into python/FLASC by Paul Fleming and Eric Simley of NREL referring to the above publication by Stoyan Kanev.  The method is named `expected_power_analysis` within FLASC to denote its calcuation of a type of expected power farm power as the weighted sum of the per-bin powers and using this to calculate the change in energy production.

The approach is different from the above approaches in several ways (refer to [AWC validation methodology](https://publications.tno.nl/publication/34637216/LWOb3s/TNO-2020-R11300.pdf) for full description).  Several key differences are that the uncertainty of the result can be computed directly from the variance and co-variances of the turbine powers.  Additionally, the method does not normalize test powers by reference powers.  In our usage we use smaller wind speed bins and wind speed estimates (rather than nacelle anemometry) to calculate the expected power. 

Similar to above, an example using the smarteole data provides usage example, see [smarteole example 10](https://github.com/NREL/flasc/blob/main/examples_smarteole/examples_smarteole/10_uplift_with_expected_power.ipynb)

