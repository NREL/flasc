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
