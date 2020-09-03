#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

setup_requirements = []

test_requirements = []

setup(
    author="Kitware, Inc.",
    author_email="kitware@kitware.com",
    python_requires="!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Framework for evaluating active learning algorithms.",
    entry_points={"console_scripts": ["tinker=tinker.main:main"]},
    package_data={"tinker": ["py.typed"]},
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords="tinker",
    name="tinker-engine",
    packages=find_packages(include=["tinker", "tinker.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.kitware.com/darpa_learn/tinker/",
    version="0.8.0",
    zip_safe=False,
)
