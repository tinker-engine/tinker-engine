"""
JPL Interface
-------------
"""
import requests
import json
from framework.harness import Harness
from framework.dataset import ImageClassificationDataset
from framework.dataset import ObjectDetectionDataset
from pathlib import Path
import pandas as pd


class JPLInterface(Harness):
    """
    JPL Interface - This interface handles
    """
    def __init__(self,
                 apikey="",
                 url="abc"):
        """


        Args:
            apikey:
            url:
        """
        # TODO: define the data_type
        self.data_type = "full"


        # TODO: The setting of these is too confusing and VERY unclear, hardcoding these now
        apikey = "adab5090-39a9-47f3-9fc2-3d65e0fee9a2"
        # url = 'https://api-dev.lollllz.com/'
        url = 'https://api-staging.lollllz.com/'
        self.apikey = apikey
        self.headers = {'user_secret': self.apikey}
        self.url = url
        self.dataset_dir = "/mnt/b8ca6451-1728-40f1-b62f-b9e07d00d3ff/data/lwll_datasets/"

        self.task_id = ""
        self.stage_id = ""
        self.sessiontoken = ""
        self.status = dict()
        self.metadata = dict()


        # Change during evaluation on DMC servers (changes paths to eval datasets)
        self.evaluate = False
        print(self.url, url)
        r = requests.get(f"{self.url}/list_tasks", headers=self.headers)
        r.raise_for_status()
        self.task_ids = r.json()['tasks']

        self.stagenames = ['base', 'adapt']
        self.checkpoint_num = 0

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

        # prep the headers for easier use in the future.
        self.headers['session_token'] = self.sessiontoken

        # load and store the session status and metadata.
        self.status = self.get_current_status()

        self.metadata = self.get_problem_metadata()

        out_obj = json.dumps(self.metadata, indent=4)

        # TODO:  Eric, can you explain this???? What is this????
        # with open("/home/eric/foo.json", "w") as outfile:
        #     outfile.write(out_obj)

        self.problem_type = self.metadata['problem_type']

    def get_whitelist_datasets_jpl(self):
        # TODO: get the whitelist datasets for this test id from the JPL server
        whitelist = self.metadata['whitelist']
        print("get whitelist datasets:", whitelist)
        external_dataset_root = f'{self.dataset_dir}/external/'
        p = Path(external_dataset_root)
        # TODO: load both train and test into same dataset
        external_datasets = dict()
        whitelist_found = set()
        for e in [x for x in p.iterdir() if x.is_dir()]:
            name = e.parts[-1]
            # If not on whitelist
            if name not in whitelist:
                print(f'Skipping {name}, not on whitelist')
                continue

            print(f'Loading {name}')
            whitelist_found.add(name)
            for split in ['train', 'test']:
                labels = pd.read_feather(
                    e / 'labels_full' / f'labels_{split}.feather')
                e_root = e / f'{name}_full' / split
                if 'bbox' in labels.columns:
                    external_datasets[f'{name}_{split}'] = ObjectDetectionDataset(
                        self,
                        dataset_name=name,
                        dataset_root=e_root,
                        seed_labels=labels)
                else:
                    external_datasets[f'{name}_{split}'] = ImageClassificationDataset(
                        self,
                        dataset_name=name,
                        dataset_root=e_root,
                        seed_labels=labels)
        whitelist_not_found = set(whitelist) - whitelist_found
        if len(whitelist_not_found) > 0:
            print('Warning: The following items are on the whitelist but '
                  'not found on the computer: ', whitelist_not_found)
        return external_datasets

    def get_whitelist_datasets(self):
        self.get_whitelist_datasets_jpl()

    def get_budget_checkpoints(self, stage, target_dataset):
        """
        Find and return the budget checkpoints from the previously loaded metadata
        """
        if stage == 'base':
            para = 'base_label_budget_full'
        elif stage == 'adapt':
            para = 'adaptation_label_budget_full'
        else:
            raise NotImplementedError('{} not implemented'.format(self.stage_id))

        total_avaialble_labels = target_dataset.unlabeled_size
        for index, budget in enumerate(self.metadata[para]):
            if budget > total_avaialble_labels:
                self.metadata[para][index] = total_avaialble_labels
            total_avaialble_labels -= self.metadata[para][index]

        return self.metadata[para]

    def start_next_checkpoint(self, stage, target_dataset, checkpoint_num):
        """  Update status and get second seed labels if on 2nd checkpoint (counting from 1)

        Args:
            stage (str): name of stage (not used here)
            target_dataset:
            checkpoint_num:

        Returns:

        """
        if checkpoint_num == 1:
            target_dataset.get_seed_labels(None, 1)
        # Update status.
        self.status = self.get_current_status()

    def get_remaining_budget(self):
        # Make sure it's up to date
        status = self.get_current_status()
        return status['budget_left_until_checkpoint']

    def post_results(self, stage_id, dataset, predictions):
        """
        Submit prediction back to JPL for evaluation

        Args:
            predictions (dict): predictions to submit in a dictionary format

        """
        return self.submit_predictions(predictions, dataset)

    def terminate_session(self):
        # Clean up early closed session
        # wipe the session id so that it can't be inadvertently used again
        if self.sessiontoken is not None:
            self.deactivate_current_session()
            self.sessiontoken = None
            del self.headers['session_token']
        self.toolset = dict()

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
        self.status = status

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

    def get_dataset_jpl(self, stage_name, dataset_split, categories=None):
        """

        Args:
            stage_name: unused here
            dataset_split:
            categories:

        Returns:

        """
        current_dataset = self.status['current_dataset']['name']
        if dataset_split == 'eval':
            dataset_split = 'test'

        dataset_root = (f'{self.dataset_dir}/development/{current_dataset}/'
                        f'{current_dataset}_full/{dataset_split}')

        name = current_dataset + '_' + dataset_split

        if self.metadata['problem_type'] == "image_classification":
            return ImageClassificationDataset(self,
                                              dataset_root=dataset_root,
                                              dataset_name=name,
                                              categories=categories)
        else:
            return ObjectDetectionDataset(self,
                                          dataset_root=dataset_root,
                                          dataset_name=name,
                                          categories=categories)

    def get_dataset(self, stage_name, dataset_name, categories=None):
        """ Load a dataset

        Args:
            stage_name:
            dataset_name:
            categories:

        Returns:

        """
        # TODO: right now assume jpl.  Need to make an automated way to tell if
        #       coco or jpl dataset.  Also, create func for loading coco
        return self.get_dataset_jpl(stage_name, dataset_name, categories=None)

    def get_seed_labels(self, dataset_name, num_seed_calls):
        """
        Get the seed labels for the dataset from JPL's server.

        Args:
            dataset_name (str): Not used here
            num_seed_calls (int): number of seed labeled level (either 0 or 1)
                necessitated by the secondary_seed_labels in the second checkpoint

        Returns:
            list[tuple[str, str]]: the initial seed labels
                a list of [filename, label] elements
        """
        if num_seed_calls == 0:
            call = 'seed_labels'
        elif num_seed_calls == 1:
            call = 'secondary_seed_labels'

        r = requests.get(f"{self.url}/{call}", headers=self.headers)
        r.raise_for_status()
        seed_labels = r.json()
        return seed_labels['Labels']

    def get_more_labels(self, fnames, dataset_name):
        """
        Query JPL's API for more labels (the active learning component).

        Danger:
            Neither JPL nor this function protects against
            relabeling images that are already labeled.
            These relabeling requests count against your budget.

        Args:
            fnames (list[str]): filenames for which you want the labels
            dataset_name (str): dataset name (used in local for lookup)

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

    def submit_predictions(self, predictions, dataset):
        """
        Submit prediction back to JPL for evaluation

        Args:
            predictions (dict): predictions to submit in a dictionary format
            dataset (framework.dataset) for which you are submitting labels

        """

        predictions = dataset.format_predictions(
            predictions[0], predictions[1])

        r = requests.post(f"{self.url}/submit_predictions",
                          json={'predictions': predictions},
                          headers=self.headers)
        r.raise_for_status()
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
            info = json.dumps(self.get_current_status(), indent=4)
        else:
            info = json.dumps(self.status, indent=4)
        return '\n'.join(['Problem/Task Status:', info, ''])

    def format_task_metadata(self):
        """
        Returns formatted string of the task/problem metadata

        Returns:
              str: Formatted String of Metadata
        """
        self.metadata = self.get_problem_metadata()
        info = json.dumps(self.metadata, indent=4)
        return '\n'.join(['Problem/Task Metadata:', info, ''])

    def __repr__(self):
        """

        Returns:
            str: Formatted String of metadata and status
        """

        return self.format_task_metadata() + '\n' + self.format_status()

    def deactivate_all_session(self):
        """ Clean up by deactivating all active sessions
        Only do this by and if necessary to clean up

        Returns:
            None
        """

        r = requests.get(f"{self.url}/list_active_sessions", headers=self.headers)
        r.raise_for_status()
        active_sessions = r.json()['active_sessions']
        for act_sess in active_sessions:
            r = requests.post(f"{self.url}/deactivate_session",
                              json={'session_token': act_sess},
                              headers=self.headers)
            r.raise_for_status()
            print(r.json())

    def deactivate_current_session(self):
        """ Clean up by deactivating a session that was canceled early

        Returns:
            None
        """

        r = requests.get(f"{self.url}/list_active_sessions", headers=self.headers)
        r.raise_for_status()
        active_sessions = r.json()['active_sessions']
        if self.sessiontoken in active_sessions:
            print("Deactivated improperly ended session")
            r = requests.post(f"{self.url}/deactivate_session",
                              json={'session_token': self.sessiontoken},
                              headers=self.headers)
            r.raise_for_status()
            print(r.json())
        else:
            print("Session Properly Deactivated")






