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
        # For now, the derrived class must define the following:
        # self.stagenames   []
        # self.task_ids     []
        pass

    @abc.abstractmethod
    def initialize_session(self, task_id):
        """
        Start a new session.
        Args:
            task_id:    one of the task ids in the list provided by get_task_ids()
        """
        #TODO: create amechanbism to reset self.stagenames.
        raise NotImplementedError

    @abc.abstractmethod
    def terminate_session(self):
        """
        End a session
        Args:
            none
        """
        raise NotImplementedError

    @abc.abstractmethod
    def post_results(self, stage_id, dataset, predictions):
        """
        Process the given results and produce statistics about the performance.
        Args:
            stage_id:       The ID of the stage for which the results are being produced
            dataset:        The dataset that was used for evaluation.
            predictions:    The predicted results when using the given dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_seed_labels(self, dataset):
        """
        get seed labels for the given dataset. These labels do not count against any budgets
        Args:
            dataset:    The dataset to get the seed labels for
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_more_labels(self, dataset, filenames):
        """
        get labels from the dataseet for the given filenames
        Args:
            dataset:    The dataset to retreive labels for
            filenames:  The names of the files that should be labeled
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_whitelist_datasets(self):
        """
        get a list if datasets that are permitted for use
        Args:
            none
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dataset(self, stage_name, dataset_subname, categories=None):
        """
        retreive the appropriate dataset for the given stage and requested naame
        Args:
            stage_name:         The name of the stage to which the dataset belongs, each
                                stage has associated datasets which can be unique to that
                                stage
            dataset_subname:    The name of the individual datset in the given stage.
                                Each stage can have multiple datasets which are
                                differentiated by name
            categories:         The initial categories that are already expected to
                                exist in the dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def start_next_checkpoint(self, stage_name, target_dataset):
        """
        setup and begin handling for a checkpoint. Starting a checkpoint allows for
        performance and accuracy measurements of each checkpoint. A checkpoint is
        effectively ended when post_results() is called.
        Args:
            stage_name:     The name of the stage to start the checkpoint for. If
                            the stage is the same as the last time start_next_checkpoint
                            was called, then this will begin the next checkpoint in that
                            stage, otherwise it wil start with the first checkpoint in
                            the stage.
            target_dataset: This is the dataset that any budgets apply to. This is needed
                            so that the harness can keep track of the budget correctly.
        """
        raise NotImplementedError


    @abc.abstractmethod
    def get_budget_checkpoints(self, stage, target_dataset):
        """
        retreive the list of checkpoint for the given stage
        Args:
            stage:          The name of the stage to get the budgets for
            target_dataset: The dataset that will be used to get labels for these
                            checkpoints. This is needed to ensure that the budgets do not
                            exceed the availabel labels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_remaining_budget(self):
        """
        get the amount of budget remaining on the current checkpoint
        Args:
            none
        """
        raise NotImplementedError


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



