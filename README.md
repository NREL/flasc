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

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


## WETO software

FLASC is primarily developed with the support of the U.S. Department of Energy and is part of the [WETO Software Stack](https://nrel.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [Controls and Analysis Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#wind-farm-controls-and-analysis)


## Installation

We recommend installing this repository in a separate virtual environment.
After creating a new virtual environment, clone this repository to your local
system and install it locally using ``pip``. The command for this is ``pip install -e flasc``.

If installing for develop, follow the developer [install instructions](https://nrel.github.io/flasc/installation.html)

## Documentation

Documentation is provided via the included examples folders as well as [online documentation](https://nrel.github.io/flasc/).

## Engaging on GitHub

FLASC leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NREL/flasc/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NREL/flasc/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/39): Include current and future work on a timeline and assign a person to "own" it

Your feedback is crucial in this environment, as it helps identify areas for enhancement, resolve issues, and ensure the project meets the needs of its users. By sharing your insights and suggestions, you contribute to the project's evolution and success.

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NREL/flasc/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NREL/flasc/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NREL/flasc/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.

# License

BSD 3-Clause License

Copyright (c) 2024, Alliance for Sustainable Energy LLC, All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
