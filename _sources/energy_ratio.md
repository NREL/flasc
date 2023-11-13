---
output:
  md_document:
    variant: markdown_github
bibliography: references.bib
---


# Energy ratio analysis

The energy ratio metric is defined as the energy yield of a test turbine
relative to the energy yield of one or multiple reference turbines for
a particular wind rose, e.g., for annual operation. See
[@Doekemeijer2022a] and [@Bay2022a] for a practical
example of how the energy ratios can be used in the analyses of
historical SCADA data of three offshore wind farms and their comparison
to an engineering wind farm model.

# Examples demonstrating energy ratio usage

Starting in FLASC v1.4, the energy ratio calculation was reconfigured around a [polars](https://www.pola.rs/) back-end and new examples were put in place to demonstrate usage.

Key syntax for computing energy ratios is provided in the examples:

 - [Energy Ratio Syntax](../examples_artificial_data/03_energy_ratio/00_demo_energy_ratio_syntax.ipynb)
 - [Energy Ratio Options](../examples_artificial_data/03_energy_ratio/01_demo_energy_ratio_options.ipynb)

 With the remaining examples in examples_artificial_data/03_energy_ratio providing more detail on specific use cases.

 Within the set of analysis of the smarteole dataset are also exmaples of using flasc's energy ratio calculations.  See for example:

 - [Baseline Energy Ratio Analysis](../examples_smarteole/05_baseline_energy_ratio_analysis.ipynb)
 - [Wake Steering Energy Ratio Analysis](../examples_smarteole/06_wake_steering_energy_ratio_analysis.ipynb)