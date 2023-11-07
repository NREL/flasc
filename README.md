# FLORIS-Based Analysis for SCADA Data (FLASC)


**Note:** Further documentation is available at **https://nrel.github.io/flasc/**

## Description

FLASC provides a **rich suite of analysis tools** for SCADA data filtering, analysis, 
wind farm model validation, field experiment design, and field experiment monitoring. 

The repository is centrally built around NRELs in-house ``FLORIS`` wind farm model, available at
**https://github.com/nrel/floris**. FLASC also largely relies on the ``energy ratio``, among others, 
to quantify wake losses in synthetic and historical data, to perform turbine northing calibrations, 
and model parameter estimation.

For technical questions or concerns, please email paul.fleming@nrel.gov.


[![pages-build-deployment](https://github.com/NREL/flasc/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/NREL/flasc/actions/workflows/pages/pages-build-deployment)

[![Automated tests & code coverage](https://github.com/NREL/flasc/actions/workflows/continuous-integration-workflow.yaml/badge.svg)](https://github.com/NREL/flasc/actions/workflows/continuous-integration-workflow.yaml)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

We recommend installing this repository in a separate virtual environment.
After creating a new virtual environment, clone this repository to your local
system and install it locally using ``pip``. The command for this is ``pip install -e flasc``.
    
## Documentation

Documentation is provided via the included examples folders as well as [online documentation](https://nrel.github.io/flasc/).

## Engaging on GitHub

FLORIS leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NREL/flasc/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NREL/flasc/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/39): Include current and future work on a timeline and assign a person to "own" it

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NREL/flasc/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NREL/flasc/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NREL/flasc/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   **http://www.apache.org/licenses/LICENSE-2.0**

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
