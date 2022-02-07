Data analysis
-------------
Once the SCADA data has been filtered, outliers have been removed and the
turbine signals have been calibrated to true north, they can be
straightforwardly used in the analysis methods implemented in FLASC.
Several of such methods are described next.

Energy ratio analysis
=====================
The energy ratio metric is defined as the energy yield of a test turbine
relative to the energy yield of one or multiple reference turbines for
a particular wind rose, e.g., for annual operation. See
:cite:`Doekemeijer2022a` and :cite:`Bay2022a` for a practical
example of how the energy ratios can be used in the analyses of
historical SCADA data of three offshore wind farms and their comparison
to an engineering wind farm model.

This repository contains two example functions concerning the calculation of
the energy ratios of various turbines in a wind farm. The first example
is ``energy_ratio_for_single_df.py``. This is described
in several steps.

+++++++++++++++++++++++++++++++++++++++++
Energy ratio curve for a single dataframe
+++++++++++++++++++++++++++++++++++++++++

Calculating the energy ratios for a dataset is mostly streamlined by the FLASC
repository. Several lower-level decisions must be made that can have smaller
to larger impacts on the energy ratio values. The example file
``energy_ratio_for_single_df.py`` is broken down and explained here.

Firstly, one must define a reference wind direction. This wind direction
must be northing-calibrated and the measurement should be valid
(i.e., should not have jumps in its northing calibration throughout the data).
The reference wind direction may be the wind direction of the test turbine, or
the average wind direction of multiple turbines neighboring the test turbine,
or even be the average wind direction in the farm (typically valid for small
farms). For larger wind farms, the reference wind direction often comes from
one or multiple turbines neighboring the test turbine, to minimize any
undesired correlations between the reference wind direction and the power
production of the test turbine. In this simple example case, we assign the
reference wind direction as being the average wind direction of all turbines
in the farm.

 .. code-block:: python

    df = dfm.set_wd_by_all_turbines(df)

If desired, we can cut down the data to a specific range of wind directions.
In the example script, we limit our range of interest to wind directions
between 20 and 90 degrees.

 .. code-block:: python

    df = dfm.filter_df_by_wd(df=df, wd_range=[20., 90.])
    df = df.reset_index(drop=True)

While the reference wind direction has been selected, the reference wind speed
has not yet been selected. We can select the reference wind speed as the average
wind speed from all upstream turbines within a 5 kilometer radius of the test
turbine, for example. However, for a small wind farm like in this example,
we can simply assign the reference wind speed as the mean wind speed of all
upstream turbines using ``set_ws_by_upstream_turbines()``. Therefore, we must
first calculate which turbines are upstream and for which wind directions,
which we do using the function ``get_upstream_turbs_floris()``. Then, we
extract the right measurements and average them accordingly. Similarly, we
must define a reference power production, by which the test turbine's power
production will be divided. Knowing which test turbine we are considering,
and knowing which turbines we want to normalize to, we can assign the
reference power production by ``set_pow_ref_by_turbines()``. Common
alternatives for the reference power are ``set_pow_ref_by_upstream_turbines()``
and ``set_pow_ref_by_upstream_turbines_in_radius()``, to assign the reference
power production according to all upstream turbines (within a certain radius
of the test turbine).

.. code-block:: python

   df_upstream = fsatools.get_upstream_turbs_floris(fi)
   df = dfm.set_ws_by_upstream_turbines(df, df_upstream)
   df = dfm.set_pow_ref_by_turbines(df, turbine_numbers=[0, 6])


Further, we must specify how we will bin the data. Namely, we bin the data
according to the reference wind direction measurement and a reference wind
speed measurement. Furthermore, we can have a larger wind direction binning
width than the binning step size to enable overlap, as also exemplified in
:cite:`Doekemeijer2022a`.  We can additionally specify a bootstrap sample
size, in case we want to calculate uncertainty bounds on the energy ratios.
For that, we would increase the value from ``N=1`` to, for example, ``N=50``.
Note that increasing ``N`` does significantly slows down the computation.

.. code-block:: python

    # # Initialize energy ratio object for the dataframe
    era = energy_ratio.energy_ratio(df_in=df, verbose=True)

    # Get energy ratio without uncertainty quantification
    era.get_energy_ratio(
        test_turbines=[1],
        wd_step=2.0,
        ws_step=1.0,
        wd_bin_width=3.0,
        N=1,
    )
    fig, ax = era.plot_energy_ratio()

This should produce the following result:

.. image:: images/example_single_df_energy_ratio.png
   :scale: 50 %
   :alt: alternate text
   :align: center



+++++++++++++++++++++++++++++++++++++++++++
Energy ratio curves for multiple dataframes
+++++++++++++++++++++++++++++++++++++++++++
In continuation of the single-dataframe energy ratio analysis, we can analyze
and cross-compare multiple energy ratios. This is exemplified in the example
``compare_energy_ratios_between_dfs.py``. It works largely in a similar manner
as the example with a single dataframe. The main difference is that now
the ``energy_ratio_suite`` class is used. Masking is also slightly different,
and can now be done through the ``energy_ratio_suite`` class directly.

.. code-block:: python

    fsc = energy_ratio_suite.energy_ratio_suite()
    fsc.add_df(df, 'Original data')
    fsc.add_df(df2, 'Data with wd bias of 7.5 degrees')

    # Now we mask the datasets to a specific wind direction subset, e.g.,
    # to 20 deg to 90 deg.
    fsc.set_masks(wd_range=[20., 90.])

    # Calculate the energy ratios for test_turbines = [1] for the masked
    # datasets with uncertainty quantification using 50 bootstrap samples
    fsc.get_energy_ratios(
        test_turbines=[1],
        wd_step=2.0,
        ws_step=1.0,
        N=50,
        percentiles=[5., 95.],
        verbose=False
    )
    fsc.plot_energy_ratios(superimpose=True)

By default, the ``energy_ratio_suite`` class also redistributes dataframes
which differ in underlying wind direction and wind speed distributions. These
dataframes are resampled onto a common wind rose, so that their energy ratios
can be compared. 

The ``energy_ratio_suite`` class is typically used to compare the dataframe
of SCADA data to the dataframes of model predictions for the same data. This
allows the user to compare, for example, FLORIS model predictions to SCADA
data. Different instantiations of the FLORIS model can be analyzed by creating
multiple dataframes and all inserting them into the suite object using the
function ``add_df()``. An example of this usage is shown in
``a_08_plot_energy_ratios.py`` in the ``examples/raw_data_processing/``
folder.


++++++++++++++++++++++++++++++++++
Comprehensive table-based analysis
++++++++++++++++++++++++++++++++++

The ``energy_ratio_suite`` described above has an additional function which
provides significant depth into the energy ratios under various scenarios.
This is the table analysis method, which generates a detailed Excel sheet
showcasing the energy ratios under various inflow conditions and in different
bins. Continuing on the example of the previous section, a user can generate
such an Excel file using the ``fsc.export_detailed_energy_info_to_xlsx()``
function. Usage of this function is exemplified in
``energy_table_for_two_df.py``, in the folder ``examples/table_analysis/``.
The function produces an Excel sheet which looks as follows:

.. image:: images/example_table_analysis.png
   :scale: 50 %
   :alt: alternate text
   :align: center

Each set of rows displays information about a single wind direction bin.
Information displayed for each wind direction bin is, for each wind speed
bin therein, the bin counts for the two dataframes (e.g., experimental data
and model-predicted data, or experimental baseline-operation data and
experimental optimal-operation data), the mean wind speed in each bin,
the mean turbulence intensity in each bin (if available in data), the
reference power production, the test power production, and the energy
ratios. Each set of rows is also accompanied by a flow field on the left
displaying the wind farm layout and the nominal wake situation for this
wind direction bin. All of this provides useful insights into, for example,
for what wind direction and wind speed bins the model-predicted dataset
significantly deviates from the experimental data.

.. seealso:: `Return to table of contents <index.html>`_ 