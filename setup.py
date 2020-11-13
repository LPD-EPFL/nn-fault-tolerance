#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ft",
    version=0.1,
    description="Fault Tolerance for Neural Networks in the Continuous Limit",
    author="El-Mahdi El-Mhamdi, Rachid Guerraoui, Andrei Kucharavy, Sergei Volodin",
    author_email="sergei.volodin@epfl.ch",
    python_requires=">=3.7.0",
    url="https://github.com/LPD-EPFL/ProbabilisticFaultToleranceNNs",
    packages=["fault_tolerance"],
    package_dir={},
    package_data={},
    # We have some non-pip packages as requirements,
    # see requirements-build.txt and requirements.txt.
    install_requires=[],
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
