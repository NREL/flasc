See https://nrel.github.io/flasc/ for documentation

____ model_fitting ____

This contains a preliminary implementation of tuning methods for FLORIS to SCADA (floris_tuning.py).
The code is focused on methods for the Empirical Guassian wake model and is
based on contributions from Elizabeth Eyeson, Paul Fleming (paul.fleming@nrel.gov)
Misha Sinner (michael.sinner@nrel.gov) and Eric Simley at NREL, and Bart
Doekemeijer at Shell, as well as discussions with Diederik van Binsbergen at
NTNU.

Please treat this module as a beta implementation.

We are planning to extend the capabilities of the model_tuning module in coming
version releases. If you are interested in contributing to this effort, please
reach out to Paul or Misha via email. Planned improvements include:
- Streamlining of processes and code
- Added flexibility for implementing other loss functions
- Consolidation and alignment with cosine power loss exponent fitting
  (see estimate_cos_pp_fit method in turbine_analysis/yaw_pow_fitting.py)
- Possible accelerated model fitting by refinement of swept parameters
- Methods for fitting multiple parameters simultaneously
