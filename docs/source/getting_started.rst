Getting started
-----------------

This section describes how the flasc repository is intended to be used, and
presents several example scripts that the user can use to become familiar
with this toolbox. Based on these example codes, it should become
straightforward how one can adapt and apply the flasc tools on their data
of interest.

General data analysis procedure
===============================
A common data analysis procedure using FLASC is as follows:

+++++++++++++++++++++++++++++++
Retrieval of the raw SCADA data
+++++++++++++++++++++++++++++++
The raw SCADA data is retrieved from an external location (e.g., FTP
server or SQL database). This step is not part of the FLASC toolbox because
it is very specific to the data location and format. It is up to the user to
download the data and format it in a Python-accessible manner (e.g., in a
.csv file).

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Format data to a wide table format (if necessary)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SCADA data can be made available in various forms. The data may already be
sampled on a common time vector, with e.g., measurements of all turbines every
10 minutes. In other situations, it may be that data is logged in a "long"
format, and where measurements are only reported if the signal has actually
changed. An example of such a data format is:

.. csv-table:: Example :rst:dir:`Tall data format for a three turbine wind farm`
   :header: "Date", "Turbine", "Variable", "Value"

   "2020-01-01 12:15:01", "A1", "NacWindSpeed", "5.5"
   "2020-01-01 12:15:01", "A2", "NacWindSpeed", "4.5"
   "2020-01-01 12:15:02", "A3", "NacWindSpeed", "7.9"
   "2020-01-01 12:15:05", "A2", "NacWindSpeed", "4.1"
   "2020-01-01 12:15:07", "A1", "NacWindSpeed", "5.9"
   "2020-01-01 12:15:11", "A3", "NacWindSpeed", "7.2"

The "long" data format is comon with high-resolution data (e.g., 1Hz), to
reduce data storage needs on the logging side. The FLASC analysis tools
require data to be in a wide table format. This means that it is up to
the user to resample the data for every wind turbine onto a common time
vector and format the data into a "wide" table format. The wide table format
looks as follows:

.. csv-table:: Example :rst:dir:`Wide data format for a three turbine wind farm`
   :header: "Date", "NacWindSpeed_A1", "NacWindSpeed_A2", "NacWindSpeed_A3"

   "2020-01-01 12:15:01", "5.5", "4.5", "7.3",
   "2020-01-01 12:15:02", "5.5", "4.5", "7.9"
   "2020-01-01 12:15:03", "5.5", "4.5", "7.9"
   "2020-01-01 12:15:04", "5.5", "4.5", "7.9"
   "2020-01-01 12:15:05", "5.5", "4.1", "5.9"

There is currently no example provided with FLASC that demonstrates the
process of data resampling and converting a long-formatted dataset to
a wide-formatted dataset. However, FLASC does provide tools to simplify
this table formatting, heavily relying on the internal functions in
the `Python Pandas library
<https://pandas.pydata.org/pandas-docs/stable/index.html>`_ for computational
efficiency.

Firstly, ``df_resample_by_interpolation()`` function allows the user to resample
a dataset which is sampled on an nonhomogeneous time vector to a time vector
with a consistent time step. This is useful, since long-formatted data is
often inconsequently sampled. One can also specify a maximum time gap for
data interpolation. If no data is available for a time period longer than
max_gap, the data will be assigned a ``NaN`` value.

 .. code-block:: python

   from flasc.time_operations import df_resample_by_interpolation
   df_res = df_resample_by_interpolation(
        df,
        time_array,
        circular_cols,
        interp_method='linear',
        max_gap=None,
        verbose=True
   )

This function also deals with 360-degree wrapping for nacelle headings and
wind direction measurements, as specified by the ``circular_cols`` option.

After each turbine's measurements are sampled onto a common time vector, they
can straightforwardly be appended into a single, wide-formatted table 
(which in Python is a ``pd.DataFrame``). A common way to format a long table
into a wide table, assuming they all share the same ``date`` vector, is:

 .. code-block:: python

    df = df.groupby(["date", "turbine", "variable"]).value.first().unstack()
    df = df.unstack()

    df.columns = ["{:s}_{:s}".format(c[0],c[1]) for c in df.columns]
    df = df.reset_index(drop=False)

or a variant of these commands.

+++++++++++++++
Data filtering
+++++++++++++++
With the SCADA data now in a wide-formatted Pandas DataFrame, the flasc
data filtering tools can readily be applied. The files in 
``examples/raw_data_processing`` demonstrate how the SCADA data files are
processed.

``a_00_initial_download.py``
This first file simply demonstrates how the raw data is imported. This
basically compromises of the previous two steps, being data downloading and
formatting it into a wide table format. Data is typically saved within flasc
using the 
`feather format <https://arrow.apache.org/docs/python/feather.html>`_, which
is known for its excellent IO speed and its efficient storage, being often a
factor 10 smaller than a similar .csv file.

``a_01_to_common_format_df.py``
This script renames the columns in the Pandas DataFrame to the conventional
variable namings that we use within flasc. We use the following naming
convention:

- The turbine wind speed in [m/s] is denoted by ``ws``.
- The turbine power production in [W] is denoted by ``pow``.
- The wind direction measured by each turbine, between 0 and 360 deg, is denoted by ``wd``.
- The turbine nacelle heading, between 0 and 360 deg, is denoted by ``yaw``.
- The turbine vane angle, between -180 and +180 deg, is denoted by ``vane``.

Each variable is appended with a three-digit identifier for the turbine. For
example, ``ws_000`` refers to the measured wind speed of the first turbine in
the wind farm. Out of these five variables, ``ws`` and ``pow`` are always
required, and either ``wd`` or ``yaw`` is necessary for most of the energy
ratio analyses.

``a_02_basic_filters.py``
This script identifies obvious data outliers. The criteria for this are
assigned by ``conds``. The example file assigns the following conditions as
being obviously faulty:

 .. code-block:: python

    conds = [
        ~df["is_operation_normal_{:03d}".format(ti)],
        df["ws_{:03d}".format(ti)] <= 0.0,
        df["pow_{:03d}".format(ti)] <= 0.0,
    ]

where the column ``is_operation_normal_000`` would refer to a turbine's
self-reported signal that identifies its operational state as normal or not.
Further, negative wind speed and power measurements are identified as faulty.


Data formatting conventions
===========================