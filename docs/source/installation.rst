Installation
-----------------
FLASC is currently not available as a package on any repository manager.
Instead, it must be installed by the user by cloning the GitHub repository.

To download the source code, use `git clone`. Then, add it to
your Python path with the "local editable install" through `pip`.

.. code-block:: bash

    # Download the source code.
    git clone https://github.com/NREL/flasc.git

    # Install into your Python environment
    pip install -e flasc

If everything is configured correctly, any changes made to the source
code will be available directly through your local Python. Remember
to re-import the FLASC module when changes are made if you are working
in an interactive environment like Jupyter.

In terms of dependencies, the flasc toolbox currently relies on floris
v2.4, and is not yet compatible with floris v3.0rc1.


.. seealso:: `Return to table of contents <index.html>`_ 