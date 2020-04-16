import abc
import os
import sys
import requests
import json
import inspect
from dataset import JPLDataset


class JPLInterface:
    def __init__(self, apikey = "", url = ""):
        #TODO: define the data_type
        self.data_type = "full"
        
        self.apikey = apikey
        self.headers = {'user_secret': self.apikey}
        self.url = url

        self.task_id = ""
        self.stage_id = ""

        #TODO: make this a parameter
        self.dataset_dir = "/mnt/b8ca6451-1728-40f1-b62f-b9e07d00d3ff/data/lwll_datasets"

        # Change during evaluation on DMC servers (changes paths to eval datasets)
        self.evalute = False

    def getTaskIDs(self):
        print ("getTestIDs")
        r = requests.get(f"{self.url}/list_tasks", headers=self.headers)
        r.raise_for_status()
        return r.json()['tasks']

    def initializeSession(self, task_id):
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

    def getWhitelistsets(self):
        #TODO: get the whitelist datasets for this test id from the JPL server
        print("getWhitelistsets")
        pass

    def getBudgetCheckpoints(self):
        if self.stage_id == 'base':
            para = 'base_label_budget'
        elif self.stage_id == 'adapt':
            para = 'adaptation_label_budget'
        else:
            raise NotImplementedError('{} not implemented'.format(self.stage_id))
        return self.metadata[para]

    def getBudgetUntilCheckpoints(self):
        # TODO: return info from session status
        # this can be loaded from self.metadata
        print("getBudgetCheckpoints")

        pass

    def getEvaluationDataSet(self):
        #TODO: return the dataset to be used for evaluating the run
        print("getEvaluationDataSet")
        pass

    def postResults(self, predictions):
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

    def terminateSession(self):
        # end the current session with the JPL server
        # no formal communication with the JPL server is needed to end the session.
        # wipe the session id so that it can't be inadvertently used again
        self.sessiontoken = None
        try:
            del self.headers['session_token']
            self.toolset = dict()
        except KeyError:
            pass
        
        print("terminateSession")
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

    def get_problem_metadata(self):
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
            f"{self.url}/task_metadata/{self.task_id}",
            headers=self.headers)
        r.raise_for_status()
        metadata = r.json()['task_metadata']
        return metadata

    def getTargetDataset(self):

        # # TODO: Remove this logic once JPL can take dynamic directory specifications
        # broken_dataset_dname = self.status['current_dataset']['data_url']
        # broken_dataset_dname = broken_dataset_dname.replace('\\', '/').split('/')
        # broken_dataset_dname = broken_dataset_dname[1:]
        # broken_dataset_dname[0] = ''
        # broken_dataset_dname[1] = self.dataset_dir
        current_dataset = self.status['current_dataset']['name']
        if self.evalute:
            dataset_root = f'{self.dataset_dir}/evaluate/{current_dataset}/{current_dataset}_{self.data_type}/train'
        else:
            dataset_root = f'{self.dataset_dir}/development/{current_dataset}/{current_dataset}_{self.data_type}/train'

        return JPLDataset(self,
                          dataset_root=dataset_root,
                          baseDataset=self.stage_id == 'base')

    def get_seed_labels(self):
        """
        Get the seed labels for the dataset from JPL's server.

        Returns:
            list[tuple[str, str]]: the initial seed labels
                a list of [filename, label] elements
        """
        import ipdb
        ipdb.set_trace()
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



