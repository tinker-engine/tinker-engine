"""
.. _problem.py:

problem.py
==========

This code is managed by Kitware.  This doesn't have to change between the
tasks.  This code interacts with JPL's API during the problem and all the
problem/task metadata while running.  This file contains all of the
interactions with JPL's API.

"""

import pandas as pd
import requests
import json
import warnings

class LwLL(object):
    """
    Learning with Less Labels (LwLL) Class which defines the problem and
    interacts with JPL's API.  You will not need to edit this class as all.

    Attributes:
        headers (dict): the initial headers to communicate with JPL API
        problem_id (str): the name of the problem
        task_metadata (dict): The information about the task at hand.
            Look at :func:`_get_problem_metadata` for more information and
            an example
        status (dict[str, str]|dict[str,int]|dict[str, list}): Status of the
            current problem/task.  Look at :func:`get_current_status` for more
            information and an example.
        problem_json (dict): the problem metadata; e.g., `'problem_test'`
        secret (str): the team's secret used for validating communication
            to JPL's server.
        dataset_dir (str): the dataset location, specified in `input.json`
        verbose (bool): whether to print information about the problem

    """

    def __init__(self, secret,
                 url,
                 problem_id,
                 data_type='full',
                 dataset_dir='/datasets'):
        """
        Init class for LwLL problem class it will:

            - Get the problem/task metadata
            - Start a Session with JPL Sever

        Args:
            secret (str): the team's secret used for validating communication
                to JPL's server.
            url (str): the URL to JPL's server
            problem_id (str): the name of the problem you are working on
            data_type (str): if you are doing the full dataset or just a sample.
                 The two choices are ``"full"`` or ``"sample"``.
            dataset_dir (str): the dataset location, specified in `input.json`
        """
        self.secret = secret
        self.url = url
        self.headers = {'user_secret': self.secret}

        self.problem_id = problem_id

        # #########  Get the problem meta data
        self.problem_json = self._get_problem_metadata
        self.task_metadata = self.problem_json['task_metadata']

        # #########  Create a Session
        self.session_token = self._get_session_token(data_type)
        #  Update header with session token
        self.headers = {'user_secret': self.secret,
                        'session_token': self.session_token}

        self.status = self.get_current_status
        self.format_status()

        self.dataset_dir = dataset_dir

    def download_dataset(self):
        """
        Download the Datasets.  Will only give warning if Learn Framework is not
        installed

        Note:
            If this fails, it's because you haven't installed the learn framework.
            Please look at the readme in the project root for more details on how to
            run this code
        """
        # ########## Download dataset here if it is not available

        try:
            from learn_framework.download import CLI
            cli = CLI()
            for dname in [self.task_metadata['base_dataset'],
                          self.task_metadata['adaptation_dataset']]:
                if self.dataset_dir:
                    cli.download_data(dname, dataset_dir=self.dataset_dir)
                else:
                    cli.download_data(dname)

        except ImportError:
            warnings.warn('Cannot download code since cannot import '
                             'learn_framework.\n'
                             'Did you pip install the framework?\n'
                             'Not an issue if datasets already downloaded.')

    @property
    def _get_problem_metadata(self):
        """
        Get the task metadata from JPL's server.
        An example: ::

            {
                "adaptation_can_use_pretrained_model": false,
                "adaptation_dataset": "mnist",
                "adaptation_evaluation_metrics": [
                    "accuracy"
                ],
                "adaptation_label_budget": [
                    1000,
                    2000,
                    3000
                ],
                "base_can_use_pretrained_model": true,
                "base_dataset": "mnist",
                "base_evaluation_metrics": [
                    "accuracy"
                ],
                "base_label_budget": [
                    3000,
                    6000,
                    8000
                ],
                "problem_type": "image_classification",
                "task_id": "problem_test_image_classification"
            }

        """
        r = requests.get(
            f"{self.url}/task_metadata/{self.problem_id}",
            headers=self.headers)
        r.raise_for_status()
        self.problem_json = r.json()
        return self.problem_json

    def _get_session_token(self, data_type):
        """
        Get the session token from JPL's server.  This should only be run once
        per task. The session token defines the instance of the problem being run.

        Args:
            data_type (str): If you are doing the full dataset or just
                a sample.  Either "full" or "sample".

        Returns:
            str: session token of problem/task

        """
        r = requests.get(
            f"{self.url}/auth/get_session_token/{data_type}/{self.problem_id}",
            headers=self.headers)
        r.raise_for_status()

        return r.json()['session_token']

    @property
    def get_current_status(self):
        """
        Get the current status of the session (will update status as well).
        This will tell you things like the number of images you can query
        before evaluation, location of the dataset, number of classes, etc.
        An example session: ::

            {
                "active": true,
                "budget_left_until_checkpoint": 3000,
                "current_dataset": {
                    "classes": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9"
                    ],
                    "data_url": "/datasets/lwll_datasets/mnist/mnist_full/train",
                    "dataset_type": "image_classification",
                    "name": "mnist",
                    "number_of_channels": 1,
                    "number_of_classes": 10,
                    "number_of_samples_test": 10000,
                    "number_of_samples_train": 60000,
                    "uid": "mnist"
                },
                "current_label_budget_stages": [
                    3000,
                    6000,
                    8000
                ],
                "date_created": 1580411853000,
                "date_last_interacted": 1580411853000,
                "pair_stage": "base",
                "task_id": "problem_test_image_classification",
                "uid": "iN9xVy67QBt71K9ATOqx",
                "user_name": "Berkely",
                "using_sample_datasets": false
            }

        Returns:
            dict[str, str]: status of problem/task
        """
        r = requests.get(f"{self.url}/session_status", headers=self.headers)
        r.raise_for_status()
        self.status = r.json()['Session_Status']

        return self.status

    def get_seed_labels(self):
        """
        Get the seed labels for the dataset from JPL's server.

        Returns:
            list[tuple[str, str]]: the initial seed labels
                a list of [filename, label] elements
        """
        r = requests.get(f"{self.url}/seed_labels", headers=self.headers)
        r.raise_for_status()
        seed_labels = r.json()
        return seed_labels['Labels']

    def get_more_labels(self, fnames):
        """
        Query JPL's API for more labels (the active learning component).

        Danger:
            Neither JPL nor this function protects against
            relabeling images that are already labeled.
            These relabeling requests count against your budget.

        Args:
            fnames (list[str]): filenames for which you want the labels

        Return:
            list: (list[tuple(str,str)]): newly labeled image filenames and classes
        """
        r = requests.post(f"{self.url}/query_labels",
                          json={'example_ids': fnames},
                          headers=self.headers)
        r.raise_for_status()
        new_data = r.json()

        self.status = new_data['Session_Status']
        return new_data['Labels']

    def submit_predictions(self, predictions):
        """
        Submit prediction back to JPL for evaluation

        Args:
            predictions (dict): predictions to submit in a dictionary format

        """
        r = requests.post(f"{self.url}/submit_predictions",
                          json={'predictions': predictions},
                          headers=self.headers)
        r.raise_for_status()
        self.status = r.json()['Session_Status']

        return self.status

    def format_status(self, update=False):
        """
        Update and return formatted string with the current status
        of the problem/task

        Args:
            update (bool): should the status be updated

        Returns:
              str: Formatted String of Status
        """
        if update:
            info = json.dumps(self.get_current_status, indent=4)
        else:
            info = json.dumps(self.status, indent=4)
        return '\n'.join(['Problem/Task Status:', info, ''])

    def format_task_metadata(self):
        """
        Returns formatted string of the task/problem metadata

        Returns:
              str: Formatted String of Metadata
        """
        info = json.dumps(self.task_metadata, indent=4)
        return '\n'.join(['Problem/Task Metadata:', info, ''])

    def __repr__(self):
        """

        Returns:
            str: Formatted String of metadata and status
        """

        return self.format_task_metadata() + '\n' + self.format_status()


def send_and_validate_response(url, headers, json=None):
    """ Validate http message

    Args:
        url (str): the url
        headers (dict): the header for the message e.g., ``{'user_secret': SECRET}``
        json (dict): Post informtation to send.
            If :obj:`None`, it's assumed to be a get request

    Returns:
        :class:`requests.Request`: request with the response

    Raises:
         :class:`requests.HTTPError`: if one occurred

    """
    # Get/Post Request
    if json is None:
        r = requests.get(url, headers=headers)
    else:  # Post request
        r = requests.post(url, json=json, headers=headers)

    # Throw exception if something goes wrong
    r.raise_for_status()
    return r
