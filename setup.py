

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'feather-format>=0.4.1',
    'matplotlib>=3',
    'numpy>=1.16',
    'openoa>=2.0.1',
    'pandas>=0.24',
    'pyproj>=2.1',
    'pytest>=4',
    'SALib>=1.4.0.2',
    'scipy>=1.1',
    'sklearn>=0.0',
    'streamlit>=0.89.0'
]

setup_requirements = [
    # TODO(pfleming): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='FLORIS_SCADA_ANALYSIS',
    version='0.1.0',
    description="Tools for comparing FLORIS and SCADA data.",
    long_description=readme + '\n\n' + history,
    author="Paul Fleming",
    author_email='paul.fleming@nrel.gov',
    url='https://github.com/NREL/floris_scada_analysis',
    packages=find_packages(include=['floris_scada_analysis']),
    entry_points={
        'console_scripts': [
            'floris_scada_analysis=floris_scada_analysis.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='floris_scada_analysis',
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
