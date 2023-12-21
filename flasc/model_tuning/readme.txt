Copyright 2023 NREL
Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

See https://nrel.github.io/flasc/ for documentation

____ model_tuning ____

This is a preliminary implementation of tuning methods for FLORIS to SCADA. 
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