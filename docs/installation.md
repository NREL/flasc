# Installation
-----------------
FLASC is currently not available as a package on any repository manager.
Instead, it must be installed by the user by cloning the GitHub repository.

To download the source code, use `git clone`. Then, add it to
your Python path with the "local editable install" through `pip`.

```bash
# Download the source code.
git clone https://github.com/NREL/flasc.git

# Install into your Python environment
pip install -e flasc

```

If installing FLASC with the intention to develop, some additional configuration is helpful:


Install FLASC in editable mode with the appropriate developer tools

   - ``".[develop]"`` is for the linting and code checking tools
   - ``".[docs]"`` is for the documentation building tools. Ideally, developers should also be
     contributing to the documentation, and therefore checking that the documentation builds locally.

```bash
pip install -e ".[develop, docs]"
```
Turn on the linting and code checking tools

```bash
pre-commit install
```

If everything is configured correctly, any changes made to the source
code will be available directly through your local Python. Remember
to re-import the FLASC module when changes are made if you are working
in an interactive environment like Jupyter.
