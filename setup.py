

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'floris==2.4',
    'feather-format>=0.4.1',
    'matplotlib>=3',
    'numpy<=1.20',
    'numba',
    'openoa>=2.0.1',
    'pandas>=1.3.0',
    'pyproj>=2.1',
    'pytest>=4',
    'SALib>=1.4.0.2',
    'scipy>=1.1',
    'sqlalchemy>=1.4.23',
    'streamlit>=0.89.0',
    'tkcalendar>=1.6.1',
]

setup_requirements = [
    # Placeholder
]

test_requirements = [
    # Placeholder
]

setup(
    name='flasc',
    version='0.1.0',
    description="FLASC provides a rich suite of analysis tools for SCADA data filtering & analysis, wind farm model validation, field experiment design, and field experiment monitoring.",
    long_description=readme + '\n\n' + history,
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
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
