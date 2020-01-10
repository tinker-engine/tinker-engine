#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'requests',
    'pandas',
    'boto3==1.10.22',
    'botocore==1.13.22',
    'docutils==0.15.2',
    'fire==0.2.1',
    'jmespath==0.9.4',
    'python-dateutil==2.8.0',
    's3transfer==0.2.1',
    'six==1.13.0',
    'termcolor==1.1.0',
    'urllib3==1.25.7'
]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Kitware, Inc.",
    author_email='kitware@kitware.com',
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Framework for evaluating active learning algorithms.",
    entry_points={
        'console_scripts': [
            'learn_framework=learn_framework.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='learn_framework',
    name='learn_framework',
    packages=find_packages(include=['learn_framework', 'learn_framework.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.kitware.com/darpa_learn/framework/',
    version='0.1.0',
    zip_safe=False,
)
