"""
Test Harness
---------------
"""
import abc

from framework.dataset import ImageClassificationDataset
from framework.dataset import ObjectDetectionDataset

class Harness(metaclass=abc.ABCMeta):
    """
    Harness
    """
    def __init__(self, configuration):
        #TODO: load the configuration from the given file using scripconfig
        # The derrived class must define the following:
        # self.stagenames   []
        # self.task_ids     []
        pass

    @abc.abstractmethod
    def initialize_session(self, task_id):
        #TODO:
        # the derived class must define the following:
        # self.checkpoints  [stages][target_datasets]
        raise NotImplementedError

    @abc.abstractmethod
    def terminate_session(self):
        raise NotImplementedError

    @abc.abstractmethod
    def post_results(self, stage_id, dataset, predictions):
        raise NotImplementedError

    @abc.abstractmethod
    def get_seed_labels(self, dataset):
        raise NotImplementedError

    @abc.abstractmethod
    def get_more_labels(self, dataset, filenames):
        raise NotImplementedError

    @abc.abstractmethod
    def get_whitelist_datasets(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(self, stage_name, dataset_subname, categories=None):
        raise NotImplementedError

    @abc.abstractmethod
    def start_next_checkpoint(self, stage_name, target_dataset):
        raise NotImplementedError


    @abc.abstractmethod
    def get_budget_checkpoints(self, stage, target_dataset):
        raise NotImplementedError

    @abc.abstractmethod
    def get_remaining_budget(self):
        raise NotImplementedError


    def get_stages(self):
        return self.stagenames

    def get_task_ids(self):
        return self.task_ids



