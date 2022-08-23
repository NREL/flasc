

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'floris>=3.1',
    'feather-format',
    'matplotlib',
    'numpy',
    'numba',
    'openoa',
    'pandas>=1.4.3',
    'pyproj',
    'pytest',
    'SALib',
    'scipy',
    'sqlalchemy',
    'streamlit',
    'tkcalendar',
    'seaborn'
]

setup_requirements = [
    # Placeholder
]

test_requirements = [
    # Placeholder
]

setup(
    name='flasc',
    version='1.0',
    description="FLASC provides a rich suite of analysis tools for SCADA data filtering & analysis, wind farm model validation, field experiment design, and field experiment monitoring.",
    long_description=readme,
    author="Bart Doekemeijer",
    author_email='bart.doekemeijer@nrel.gov',
    url='https://github.com/NREL/flasc',
    packages=find_packages(include=['flasc']),
    entry_points={
        'console_scripts': [
            'flasc=flasc.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='flasc',
    classifiers=[
        'Development Status :: Release',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
