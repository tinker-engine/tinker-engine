import abc
import os
import sys
import requests
import json
import inspect


class LocalInterface:
    def __init__(self, json_configuration_file):

        with open(json_configuration_file) as json_file:
            self.configuration_data = json.load(json_file)

    def get_task_ids(self):
        return self.configuration_data.keys()

    def initialize_session(self, task_id):
        self.metadata = self.configuration_data[task_id]

    def get_whitelist_datasets(self):
        #TODO:

        pass

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
        #TODO:
        pass

    def get_evaluation_dataset(self):
        # TODO: return the dataset to be used for evaluating the run
        pass

    def update_external_datasets(self):
        # TODO:
        pass

    def get_target_dataset(self, dset='train', categories=None):
        # TODO:
        pass

    def get_more_labels(self, fnames):
        # TODO:
        pass

    def get_seed_labels(self):
        # TODO:
        pass

    def post_results(self, predictions):
        #TODO:
        pass

    def get_problem_metadata(self, task_id=None):
        #TODO:
        pass

    def terminate_session(self):
        #TODO:
        pass

