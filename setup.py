#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

setup_requirements = []

test_requirements = []

def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv("CI_COMMIT_BRANCH") == 'master':
        return ""
    else:
        return get_local_node_and_date(version)

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
    use_scm_version={'local_scheme': prerelease_local_scheme},
    zip_safe=False,
)
