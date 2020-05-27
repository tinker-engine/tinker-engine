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
    JPL Interface - This interface handles the backend for integration with the
    JPL test harness.  It handles all the rest calls.
    """

    def __init__(self, apikey="", url=""):
        """ Constructs the JPL problem and get list of tasks. Args will be moved to
        config

        """
        # TODO: define the data_type
        self.data_type = "full"

        # TODO: The setting of these is too confusing and VERY unclear, hardcoding these now
        apikey = "adab5090-39a9-47f3-9fc2-3d65e0fee9a2"
        # url = 'https://api-dev.lollllz.com/'
        url = "https://api-staging.lollllz.com/"
        self.apikey = apikey
        self.headers = {"user_secret": self.apikey}
        self.url = url
        self.dataset_dir = (
            "/mnt/b8ca6451-1728-40f1-b62f-b9e07d00d3ff/data/lwll_datasets/"
        )

        self.task_id = ""
        self.stage_id = ""
        self.sessiontoken = ""
        self.status = dict()
        self.metadata = dict()

        # Change url during evaluation on DMC servers (changes paths to eval datasets)
        self.evaluate = False
        print("Using this URL:", self.url)
        r = requests.get(f"{self.url}/list_tasks", headers=self.headers)
        r.raise_for_status()
        self.task_ids = r.json()["tasks"]

        self.stagenames = ["base", "adapt"]
        self.checkpoint_num = 0

    def initialize_session(self, task_id: str) -> None:
        """
        Get the session token from JPL's server.  This should only be run once
        per task. The session token defines the instance of the problem being run.

        Args:
            task_id (str): Name of the task you would like to start

        Returns:
            none

        """
        self.task_id = task_id
        _json = {
            "session_name": "testing",
            "data_type": self.data_type,
            "task_id": task_id,
        }
        r = requests.post(
            f"{self.url}/auth/create_session", headers=self.headers, json=_json
        )
        r.raise_for_status()

        self.sessiontoken = r.json()["session_token"]

        # prep the headers for easier use in the future.
        self.headers["session_token"] = self.sessiontoken

        # load and store the session status and metadata.
        self.status = self.get_current_status()

        self.metadata = self.get_problem_metadata()

        out_obj = json.dumps(self.metadata, indent=4)

        self.problem_type = self.metadata["problem_type"]

    def get_whitelist_datasets_jpl(self):
        """ Get all the whitelisted datasets requested in the task if they are on
        disk.  Warn if they aren't on disk.

        Returns:
            dict[framework.dataset]: dictionary of datasets
        """
        whitelist = self.metadata["whitelist"]
        print("Get whitelist datasets:", whitelist)
        external_dataset_root = f"{self.dataset_dir}/external/"
        p = Path(external_dataset_root)
        # TODO: Iter whitelist rather than ones on disk.  This is fine for now.
        external_datasets = dict()
        whitelist_found = set()
        for e in [x for x in p.iterdir() if x.is_dir()]:
            name = e.parts[-1]
            # If not on whitelist
            if name not in whitelist:
                print(f"Skipping {name}, not on whitelist")
                continue

            print(f"Loading {name}")
            whitelist_found.add(name)
            for split in ["train", "test"]:
                labels = pd.read_feather(e / "labels_full" / f"labels_{split}.feather")
                e_root = e / f"{name}_full" / split
                if "bbox" in labels.columns:
                    external_datasets[f"{name}_{split}"] = ObjectDetectionDataset(
                        self, dataset_name=name, dataset_root=e_root, seed_labels=labels
                    )
                else:
                    external_datasets[f"{name}_{split}"] = ImageClassificationDataset(
                        self, dataset_name=name, dataset_root=e_root, seed_labels=labels
                    )
        whitelist_not_found = set(whitelist) - whitelist_found
        if len(whitelist_not_found) > 0:
            print(
                "Warning: The following items are on the whitelist but "
                "not found on the computer: ",
                whitelist_not_found,
            )
        return external_datasets

    def get_whitelist_datasets(self):
        self.get_whitelist_datasets_jpl()

    def get_budget_checkpoints(self, stage, target_dataset):
        """
        Find and return the budget checkpoints from the previously loaded metadata

        Args:
            stage (str): current stage
            target_dataset (framework.dataset): dataset from which you want
                the budget
        """
        if stage == "base":
            para = "base_label_budget_full"
        elif stage == "adapt":
            para = "adaptation_label_budget_full"
        else:
            raise NotImplementedError("{} not implemented".format(self.stage_id))

        total_available_labels = target_dataset.unlabeled_size
        for index, budget in enumerate(self.metadata[para]):
            if budget > total_available_labels:
                self.metadata[para][index] = total_available_labels
            total_available_labels -= self.metadata[para][index]

        return self.metadata[para]

    def start_checkpoint(self, stage, target_dataset, checkpoint_num):
        """  Update status and get second seed labels if on 2nd checkpoint
        (counting from 1)

        Args:
            stage (str): name of stage (not used here)
            target_dataset: target dataset for this checkpoint
                (used to get secondary seed labels)
            checkpoint_num: the current number of the checkpoint
                Only calls secondary seed labels

        Returns:

        """
        if checkpoint_num == 1:
            target_dataset.get_seed_labels(None, 1)
        # Update status.
        self.status = self.get_current_status()

    def get_remaining_budget(self) -> int:
        """ Update the status and then use it to return the current budget

        Returns:
            Current budget

        """
        # Make sure it's up to date
        status = self.get_current_status()
        return status["budget_left_until_checkpoint"]

    def post_results(self, stage_id, dataset, predictions):
        """
        Submit prediction back to JPL for evaluation

        Args:
            stage_id (str): unused here.
            dataset (framework.dataset): the dataset that you are predicting on
                Necessary to format the predictions correctly
            predictions (dict): predictions to submit in a dictionary format

        """
        return self.submit_predictions(predictions, dataset)

    def terminate_session(self):
        """ clean up session and close if not closed yet
        wipe the session id so that it can't be inadvertently used again

        Returns:
            None
        """

        if self.sessiontoken is not None:
            self.deactivate_current_session()
            self.sessiontoken = None
            del self.headers["session_token"]
        self.toolset = dict()

    def get_current_status(self):
        """
        Get the current status of the session.
        This will tell you things like the number of images you can query
        before evaluation, location of the dataset, number of classes, etc.

        An example session: ::

            { 'active': 'In Progress',
              'budget_left_until_checkpoint': 10,
              'budget_used': 0,
              'current_dataset': {'classes': ['0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9'],
               'data_url': '/datasets/lwll_datasets/mnist/mnist_sample/train',
               'dataset_type': 'image_classification',
               'name': 'mnist',
               'number_of_channels': 1,
               'number_of_classes': 10,
               'number_of_samples_test': 1000,
               'number_of_samples_train': 5000,
               'uid': 'mnist'},
              'current_label_budget_stages': [10, 30, 70, 165, 387, 909, 2131, 5000],
              'date_created': 1588221182000,
              'date_last_interacted': 1588221182000,
              'pair_stage': 'base',
              'session_name': 'testing',
              'task_id': 'problem_test_image_classification',
              'uid': 'k6tE4Vo2oFEuSHo7MhUE',
              'user_name': 'JPL',
              'using_sample_datasets': True}

        Returns:
            dict[str, str]: status of problem/task
        """
        r = requests.get(f"{self.url}/session_status", headers=self.headers)
        r.raise_for_status()
        status = r.json()["Session_Status"]
        self.status = status

        return status

    def get_problem_metadata(self, task_id=None):
        """
        Get the task metadata from JPL's server.

        Args:
            task_id (str): task id for which you want the metadata for.  If none
                given, assume you want the current one

        An example: ::

              "task_metadata": {
                "adaptation_dataset": "mnist",
                "adaptation_evaluation_metrics": [
                  "accuracy"
                ],
                "adaptation_label_budget_full": [
                  10,
                  110,
                  314,
                  899,
                  2569,
                  7343,
                  20991,
                  60000
                ],
                "adaptation_label_budget_sample": [
                  10,
                  30,
                  70,
                  165,
                  387,
                  909,
                  2131,
                  5000
                ],
                "base_dataset": "cifar100",
                "base_evaluation_metrics": [
                  "accuracy"
                ],
                "base_label_budget_full": [
                  100,
                  1100,
                  2078,
                  3926,
                  7416,
                  14010,
                  26467,
                  50000
                ],
                "base_label_budget_sample": [
                  100,
                  300,
                  479,
                  766,
                  1225,
                  1957,
                  3128,
                  5000
                ],
                "problem_type": "image_classification",
                "task_id": "6d5e1f85-5d8f-4cc9-8184-299db03713f4",
                "whitelist": [
                  "imagenet1k",
                  "domain_net-painting",
                  "coco2014"
                ]
              }
            }

        """
        if task_id is None:
            task_id = self.task_id
        r = requests.get(f"{self.url}/task_metadata/{task_id}", headers=self.headers)
        r.raise_for_status()
        metadata = r.json()["task_metadata"]
        return metadata

    def get_dataset_jpl(self, stage_name, dataset_split, categories=None):
        """ Get a dataset which has been downloaded from the JPL Repo and
        return a framework.dataset with that data.

        Args:
            stage_name (str): unused here
            dataset_split (str): whether get train or test data
            categories (list[str]): if test data, list of catagories.
                Otherwise None

        Returns:
            requested datasets as a framework.dataset
        """
        current_dataset = self.status["current_dataset"]["name"]
        if dataset_split == "eval":
            dataset_split = "test"

        dataset_root = (
            f"{self.dataset_dir}/development/{current_dataset}/"
            f"{current_dataset}_full/{dataset_split}"
        )

        name = current_dataset + "_" + dataset_split

        if self.metadata["problem_type"] == "image_classification":
            return ImageClassificationDataset(
                self,
                dataset_root=dataset_root,
                dataset_name=name,
                categories=categories,
            )
        else:
            return ObjectDetectionDataset(
                self,
                dataset_root=dataset_root,
                dataset_name=name,
                categories=categories,
            )

    def get_dataset(self, stage_name, dataset_name, categories=None):
        """ Load a dataset

        Wrapper right now for :meth:get_dataset_jpl

        Args:
            stage_name:
            dataset_name:
            categories:

        Returns:

        """
        # TODO: right now assume jpl.  Need to make an automated way to tell if
        #       coco or jpl dataset.  Also, create func for loading coco
        return self.get_dataset_jpl(stage_name, dataset_name, categories=None)

    def get_seed_labels(self, dataset_name, seed_level):
        """
        Get the seed labels for the dataset from JPL's server.

        Args:
            dataset_name (str): Not used here
            seed_level (int): number of seed labeled level (either 0 or 1)
                necessitated by the secondary_seed_labels in the second checkpoint

        Returns:
            list[tuple[str, str]]: the initial seed labels
                a list of [filename, label] elements
        """
        if seed_level == 0:
            call = "seed_labels"
        elif seed_level == 1:
            call = "secondary_seed_labels"

        r = requests.get(f"{self.url}/{call}", headers=self.headers)
        r.raise_for_status()
        seed_labels = r.json()
        return seed_labels["Labels"]

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
        r = requests.post(
            f"{self.url}/query_labels",
            json={"example_ids": fnames},
            headers=self.headers,
        )
        r.raise_for_status()
        new_data = r.json()

        self.status = new_data["Session_Status"]
        return new_data["Labels"]

    def submit_predictions(self, predictions: dict, dataset) -> dict:
        """
        Submit prediction back to JPL for evaluation.  Look at
        ..:meth: framework.dataset.format_predictions for formatting info

        Args:
            predictions (dict): predictions to submit in a dictionary format
            dataset (framework.dataset) for which you are submitting labels

        Returns:
            dict for status
        """

        predictions = dataset.format_predictions(predictions[0], predictions[1])

        r = requests.post(
            f"{self.url}/submit_predictions",
            json={"predictions": predictions},
            headers=self.headers,
        )
        r.raise_for_status()
        self.status = r.json()["Session_Status"]
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
        return "\n".join(["Problem/Task Status:", info, ""])

    def format_task_metadata(self):
        """
        Returns formatted string of the task/problem metadata

        Returns:
              str: Formatted String of Metadata
        """
        self.metadata = self.get_problem_metadata()
        info = json.dumps(self.metadata, indent=4)
        return "\n".join(["Problem/Task Metadata:", info, ""])

    def __repr__(self):
        """

        Returns:
            str: Formatted String of metadata and status
        """

        return self.format_task_metadata() + "\n" + self.format_status()

    def deactivate_all_session(self):
        """ Clean up by deactivating all active sessions
        Only do this by and if necessary to clean up

        Returns:
            None
        """

        r = requests.get(f"{self.url}/list_active_sessions", headers=self.headers)
        r.raise_for_status()
        active_sessions = r.json()["active_sessions"]
        for act_sess in active_sessions:
            r = requests.post(
                f"{self.url}/deactivate_session",
                json={"session_token": act_sess},
                headers=self.headers,
            )
            r.raise_for_status()
            print(r.json())

    def deactivate_current_session(self):
        """ Clean up by deactivating a session that was canceled early or do nothing

        Returns:
            None
        """

        r = requests.get(f"{self.url}/list_active_sessions", headers=self.headers)
        r.raise_for_status()
        active_sessions = r.json()["active_sessions"]
        if self.sessiontoken in active_sessions:
            r = requests.post(
                f"{self.url}/deactivate_session",
                json={"session_token": self.sessiontoken},
                headers=self.headers,
            )
            r.raise_for_status()
            print("Deactivated improperly ended session")
        else:
            print("Session Properly Deactivated")

    def get_stages(self):
        """
        return a list of the stages in the current task
        Args:
            none
        """
        return self.stagenames

    def get_task_ids(self):
        """
        return a list of tasks
        Args:
            none
        """
        return self.task_ids
