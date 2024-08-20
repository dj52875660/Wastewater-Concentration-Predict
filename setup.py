#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

setup(
    name='mx_ph',
    packages=find_packages(
        include=['mx_ph', 'mx_ph.*']
    ),
    test_suite='tests',
    version="0.1.0",
)
