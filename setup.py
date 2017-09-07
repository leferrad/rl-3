#! /usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages

setup(
    name="rl3",
    version="0.1.0",
    author="Leandro Ferrado",
    author_email="ljferrado@gmail.com",
    url="https://github.com/leferrad/rl-3",
    packages=find_packages(exclude=['examples', 'docs', 'test']),
    license="LICENSE",
    description="Reinforcement Learning raised to the third power to solve a magic cube",
    long_description=open("README.md").read(),
    install_requires=open("requirements.txt").read().split()
)