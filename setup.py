"""The setup script."""

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "flasc"
DESCRIPTION = (
    "FLASC provides a rich suite of analysis tools for SCADA data filtering & analysis, "
    " wind farm model validation, field experiment design, and field experiment monitoring."
)
URL = "https://github.com/NREL/flasc"
EMAIL = "paul.fleming@nrel.gov"
AUTHOR = "NREL National Wind Technology Center"

# What packages are required for this module to be executed?
REQUIRED = [
    "bokeh>=2, <4",
    "floris~=4.0",
    "feather-format~=0.0",
    "ipympl~=0.9",
    "matplotlib~=3.8",
    "numpy~=1.20",
    "pandas~=2.0",
    "pyproj~=3.0",
    "SALib~=1.0",
    "scipy~=1.1",
    "streamlit~=1.0",
    "tkcalendar~=1.0",
    "seaborn~=0.0",
    "polars==0.19.5",
    "ephem",
    "coloredlogs~=10.0",
]

EXTRAS = {
    "docs": {
        "jupyter-book",
        "sphinx-book-theme",
        "sphinx-autodoc-typehints",
        "sphinxcontrib-autoyaml",
        "sphinxcontrib.mermaid",
    },
    "develop": {
        "pytest",
        "pre-commit",
        "ruff",
        "isort",
    },
}

ROOT = Path(__file__).parent
with open(ROOT / "flasc" / "version.py") as version_file:
    VERSION = version_file.read().strip()

with open("README.md") as readme_file:
    README = readme_file.read()

setup_requirements = [
    # Placeholder
]

test_requirements = [
    # Placeholder
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(include=["flasc*"]),
    entry_points={"console_scripts": ["flasc=flasc.cli:main"]},
    include_package_data=True,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    license_files=("LICENSE.txt",),
    zip_safe=False,
    keywords="flasc",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    test_suite="tests",
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
