# Calculating the change in energy production

To investigate the change in energy production from a field test of a technology such as wind farm control, the FLASC repository was originally built around [energy ratio analysis](energy_ratio) with a focus on observing patterns in changes in energy production and visually comparing those with those changes expected from wake models.

FLASC now however includes three methods for quantifying the change in energy production.

## Total Uplift Power Ratio

`total_uplift_power_ratio` uses a similar input as the [energy ratio](energy_ratio) methods but returns a single value representing the total uplift, rather than uplift binned by wind direction.  The method is named power_ratio because the change in energy production is computed using the mean per-wind-condition-bin power ratios.  The change in ratio per bin is then combined with the mean base power and frequency to estimate change in energy production.

Currently the main example usage of the total uplift function is at the end of [smarteole example 06](https://github.com/NREL/flasc/blob/main/examples_smarteole/06_wake_steering_energy_ratio_analysis.ipynb).  Documentation of the function itself is available in the [API documentation](https://nrel.github.io/flasc/_autosummary/flasc.analysis.total_uplift_power_ratio.compute_total_uplift.html#flasc.analysis.total_uplift_power_ratio.compute_total_uplift).  Uncertainty of the results can be computed via bootstrapping.

The method was developed by Eric Simley and implemented by Paul Fleming and Misha Sinner of NREL.  

## Wind-Up

FLASC further includes methods for calculating change in energy production using the [wind-up](https://github.com/resgroup/wind-up) module.   [wind-up](https://github.com/resgroup/wind-up) is a tool to assess yield uplift of wind turbines developed by Alex Clerc of RES and available open-source on GitHub.  Using translation methods in the [FlascDataFrame](flasc_data_format), the methods and analysis of wind-up can be invoked from FLASC.

[smarteole example 09](https://github.com/NREL/flasc/blob/main/examples_smarteole/09_wind-up_wake_steering_uplift_analysis.ipynb), calculates the change in energy production (as in [smarteole example 06](https://github.com/NREL/flasc/blob/main/examples_smarteole/06_wake_steering_energy_ratio_analysis.ipynb)) using wind-up.

## Expected Power Analysis

The final included methodology for calculating change in energy production is the module `expected_power_analysis`.  This module implements the calculations of change in energy production described in [AWC validation methodology](https://publications.tno.nl/publication/34637216/LWOb3s/TNO-2020-R11300.pdf), by Stoyan Kanev of TNO.  The method was implemented into python/FLASC by Paul Fleming and Eric Simley of NREL referring to the above publication by Stoyan Kanev.  The method is named `expected_power_analysis` within FLASC to denote its calculation of the expected farm power as the weighted sum of the per-bin expected powers, rather than power ratios, and using this to calculate the change in energy production.

Specifically, this module computes the total uplift along with the confidence interval of the total uplift estimate by implementing Equations 4.11 - 4.29 in the abovementioned TNO report. To determine the expected wind farm power for each wind direction/wind speed bin the expected power of each individual turbine is summed for the bin. One advantage of this method is that by computing expected power at the turbine level before summing, the method does not require that all test turbines are operating normally at each timestamp. Total wind farm energy is then computed by summing the expected farm power values weighted by their frequencies of occurrence over all wind condition bins. 

The module provides two approaches for quantifying uncertainty in the total uplift. First, bootstrapping can be used similar to the first two methods. The second option approximates uncertainty in the total uplift following the approach in the abovementioned TNO report by propagating the standard errors of the expected wind farm power in each wind condition bin for the two control modes following analytic expressions derived by linearizing the total uplift formula. Benefits of this approach include higher computational efficiency compared to bootstrapping, which relies on computing the uplift for many different iterations. However, challenges with computing the required variances and covariances of wind turbine power can arise for bins with very little data (though this is accounted for automatically by approximating the missing terms).

The approach is different from the above approaches in several ways (refer to [AWC validation methodology](https://publications.tno.nl/publication/34637216/LWOb3s/TNO-2020-R11300.pdf) for full description).  First, as mentioned above, the uncertainty of the result can be computed directly from the variance and co-variances of the turbine powers instead of relying on the more computationally expensive bootstrapping approach.  Additionally, the method does not normalize the power of the test turbines by reference powers.  Therefore, the method may be more sensitive to wind speed variations within bins and other atmospheric conditions that would otherwise be partially controlled for through normalization by reference powers. To account for this sensitivity we suggest using smaller wind speed bins and wind speed estimates based on measured turbine performance (rather than nacelle anemometry) when calculating the expected power.  However, by avoiding normalization by a reference power signal, this method does not require all test turbines to be operating normally at each sample, increasing the amount of usable data, especially when large wind farms are being analyzed.

Similar to above, an example using the smarteole data provides usage example, see [smarteole example 10](https://github.com/NREL/flasc/blob/main/examples_smarteole/examples_smarteole/10_uplift_with_expected_power.ipynb)

