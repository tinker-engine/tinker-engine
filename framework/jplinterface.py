import abc
import os
import sys
import requests
import json
import inspect
from framework.dataset import JPLDataset


class JPLInterface:
    def __init__(self, apikey = "", url = ""):
        # TODO: define the data_type
        self.data_type = "full"
        
        self.apikey = apikey
        self.headers = {'user_secret': self.apikey}
        self.url = url

        self.task_id = ""
        self.stage_id = ""
        self.sessiontoken = ""
        self.status = dict()
        self.metadata = dict()

        # TODO: make this a parameter
        self.dataset_dir = ""

        # Change during evaluation on DMC servers (changes paths to eval datasets)
        self.evaluate = False

    def get_task_ids(self):
        print("getTestIDs")
        r = requests.get(f"{self.url}/list_tasks", headers=self.headers)
        r.raise_for_status()
        return r.json()['tasks']

    def initialize_session(self, task_id):
        """
        Get the session token from JPL's server.  This should only be run once
        per task. The session token defines the instance of the problem being run.

        Returns:
            none

        """
        self.task_id = task_id
        _json = {'session_name': 'testing',
                'data_type': self.data_type,
                'task_id': task_id}
        r = requests.post(
            f"{self.url}/auth/create_session",
            headers=self.headers,
            json=_json
        )
        r.raise_for_status()

        self.sessiontoken = r.json()['session_token']

        #prep the headers for easier use in the future.
        self.headers['session_token'] = self.sessiontoken

        # load and store the session status and metadata.
        self.status = self.get_current_status()

        self.metadata = self.get_problem_metadata()

        self.problem_type = self.metadata['problem_type']

    def get_whitelist_datasets(self):
        # TODO: get the whitelist datasets for this test id from the JPL server
        print("get whitelist datasets")
        from pathlib import Path
        import pandas as pd
        external_dataset_root = f'{self.dataset_dir}/external/'
        p = Path(external_dataset_root)
        # TODO: load both train and test into same dataset
        external_datasets = dict()
        for e in [x for x in p.iterdir() if x.is_dir()]:
            name = e.parts[-1]
            print(f'Loading {name}')
            for dset in ['train', 'test']:
                labels = pd.read_feather(e / 'labels_full' / f'labels_{dset}.feather')
                if 'bbox' in labels.columns:
                    dataset_type = 'object_detection'
                else:
                    dataset_type = 'image_classification'
                e_root = e / f'{name}_full' / dset
                external_datasets[f'{name}_{dset}'] = JPLDataset(self,
                          dataset_root=e_root,
                          dataset_id=f'{name}_{dset}',
                          dataset_type=dataset_type,
                          seed_labels=labels)

        return external_datasets

    def get_budget_checkpoints(self):
        """
        Find and return the budget checkpoints from the previously loaded metadata
        """
        if self.stage_id == 'base':
            para = 'base_label_budget'
        elif self.stage_id == 'adapt':
            para = 'adaptation_label_budget'
        else:
            raise NotImplementedError('{} not implemented'.format(self.stage_id))
        return self.metadata[para]

    def get_budget_until_checkpoints(self):
        # TODO: return info from session status
        # this can be loaded from self.metadata
        pass

    def get_evaluation_dataset(self):
        categories = self.toolset['target_dataset'].categories
        return self.get_target_dataset(dset='test', categories=categories)

    def post_results(self, predictions):
        """
        Submit prediction back to JPL for evaluation

        Args:
            predictions (dict): predictions to submit in a dictionary format

        """
        return self.submit_predictions(predictions)

    def terminate_session(self):
        # end the current session with the JPL server
        # no formal communication with the JPL server is needed to end the session.
        # wipe the session id so that it can't be inadvertently used again
        self.sessiontoken = None
        try:
            del self.headers['session_token']
            self.toolset = dict()
        except KeyError:
            pass
        
        pass

    def get_current_status(self):
        """
        Get the current status of the session.
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
        status = r.json()['Session_Status']

        return status

    def get_problem_metadata(self, task_id=None):
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
        if task_id is None:
            task_id = self.task_id
        r = requests.get(
            f"{self.url}/task_metadata/{task_id}",
            headers=self.headers)
        r.raise_for_status()
        metadata = r.json()['task_metadata']
        return metadata

    def get_target_dataset(self, dset='train', categories=None):
        current_dataset = self.status['current_dataset']['name']
        if self.evaluate:
            dataset_root = (f'{self.dataset_dir}/evaluate/{current_dataset}/' 
                            f'{current_dataset}_{self.data_type}/{dset}')
        else:
            dataset_root = (f'{self.dataset_dir}/development/{current_dataset}/' 
                            f'{current_dataset}_{self.data_type}/{dset}')

        return JPLDataset(self,
                          dataset_root=dataset_root,
                          dataset_id=current_dataset,
                          dataset_type=self.metadata['problem_type'],
                          categories=categories)

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

        predictions = self.toolset['eval_dataset'].format_predictions(
            predictions[0], predictions[1])

        r = requests.post(f"{self.url}/submit_predictions",
                          json={'predictions': predictions},
                          headers=self.headers)
        r.raise_for_status()
        self.status = r.json()['Session_Status']

        return self.status

    def update_external_datasets(self):
        target_name = self.toolset["target_dataset"].name
        train_id = f'{target_name}_train'
        test_id = f'{target_name}_test'
        self.toolset["whitelist_datasets"][train_id] = self.toolset["target_dataset"]
        self.toolset["whitelist_datasets"][test_id] = self.toolset["eval_dataset"]

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
        info = json.dumps(self.metadata, indent=4)
        return '\n'.join(['Problem/Task Metadata:', info, ''])

    def __repr__(self):
        """

        Returns:
            str: Formatted String of metadata and status
        """

        return self.format_task_metadata() + '\n' + self.format_status()
