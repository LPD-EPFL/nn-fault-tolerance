#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ft",
    version=0.1,
    description="Fault Tolerance for Neural Networks in the Continuous Limit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="El-Mahdi El-Mhamdi, Rachid Guerraoui, Andrei Kucharavy, Sergei Volodin",
    author_email="sergei.volodin@epfl.ch",
    python_requires=">=3.7.0",
    url="https://github.com/LPD-EPFL/ProbabilisticFaultToleranceNNs",
    packages=["fault_tolerance"],
    package_dir={},
    package_data={'fault_tolerance': ['fault_tolerance/config/*.gin']},
    install_requires=required,
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
