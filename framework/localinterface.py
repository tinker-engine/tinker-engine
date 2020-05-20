"""
Local Interface
---------------
"""

import os
import sys
import json
import inspect
from pathlib import Path
import pandas as pd
from framework.harness import Harness
from framework.dataset import ImageClassificationDataset
from framework.dataset import ObjectDetectionDataset


class LocalInterface(Harness):
    """  Local interface

    """
    def __init__(self, json_configuration_file, interface_config_path):
        """
        Initialize the object by loading the given configuration file. The configuration file
        is expected to be located relative to the interface_config_path.

        Args:
            json_configuration_file:
        """

        json_full_path = os.path.join(interface_config_path, json_configuration_file)
        print("Protocol path", json_full_path)
        if not os.path.exists(json_full_path):
            print("Given LocalInterface configuration file does not exist")
            exit(1)

        with open(json_full_path) as json_file:
            self.configuration_data = json.load(json_file)
        self.metadata = None
        self.toolset = dict()

        self.task_ids = self.configuration_data.keys()

    def initialize_session(self, task_id):
        """
        Open a new session for the given task id. This will reset old information from
        any previous sessions. Do not call this before posting results from an existing
        session or the results will not post properly.

        Args:
            task_id:

        Returns:

        """
        # clear any old session data, and prepare for the next task
        self.metadata = self.configuration_data[task_id]

        # prep the stagenames from the metadata for easier searching later.
        self.stagenames = []
        for stage in self.metadata["stages"]:
            self.stagenames.append(stage['name'])

        self.current_stage = None
        self.seed_labels = dict()
        self.label_sets = dict()
        self.label_sets_pd = dict()

    def get_whitelist_datasets_jpl(self):
        """
        Return a list of datasets that are acceptable for use in training. This will
        return all datasets present in the location indicated by the configuration option:
        "external_dataset_location".

        Returns:

        """
        print("get whitelist datasets")
        from pathlib import Path
        import pandas as pd
        external_dataset_root = f'{self.metadata["external_dataset_location"]}'
        p = Path(external_dataset_root)
        # TODO: load both train and test into same dataset
        external_datasets = dict()
        for e in [x for x in p.iterdir() if x.is_dir()]:
            name = e.parts[-1]
            print(f'Loading {name}')
            for dset in ['train', 'test']:
                labels = pd.read_feather(
                    e / 'labels_full' / f'labels_{dset}.feather')
                e_root = e / f'{name}_full' / dset
                if 'bbox' in labels.columns:
                    external_datasets[f'{name}_{dset}'] = ObjectDetectionDataset(
                        self,
                        dataset_name=name,
                        dataset_root=e_root,
                        seed_labels=labels)
                else:
                    external_datasets[f'{name}_{dset}'] = ImageClassificationDataset(
                        self,
                        dataset_name=name,
                        dataset_root=e_root,
                        seed_labels=labels)
        return external_datasets

    def get_whitelist_datasets(self):
        """
        Return a list of datasets that are acceptable for use in training. This will
        return all datasets present in the location indicated by the configuration option:
        "external_dataset_location".

        Returns:

        """
        # TODO: This function currently goes through the entire external_dataset_location
        # and returns every dataset it finds. This should be modified to
        # accept a configuration option to define the datasets that are whitelisted
        external_dataset_root = self.metadata["external_dataset_location"]
        p = Path(external_dataset_root)
        # load both train and test into same dataset
        external_datasets = dict()
        for e in [x for x in p.iterdir() if x.is_dir()]:
            name = e.parts[-1]
            print("Loading external dataset", name)
            labels = pd.read_feather(e / 'labels' / 'labels.feather')
            if 'bbox' in labels.columns:
                external_datasets[name] = ObjectDetectionDataset(
                    self,
                    dataset_root=e,
                    dataset_name=name,
                    seed_labels=labels)
            else:
                external_datasets[name] = ImageClassificationDataset(
                    self,
                    dataset_root=e,
                    dataset_name=name,
                    seed_labels=labels)

        return external_datasets

    def get_budget_checkpoints(self, stage, target_dataset):
        """
        Find and return the budget checkpoints from the previously loaded metadata.
        This uses the configuration option: "label_budget"

        Args:
            stage (str):
            target_dataset (str):
        """
        stage_metadata = self.get_stage_metadata(stage)
        if stage_metadata:
            # check the dataset to make sure that the budgets don't exceed the available
            # labels
            total_avaialble_labels = target_dataset.unlabeled_size
            for index, budget in enumerate(stage_metadata['label_budget']):
                if budget > total_avaialble_labels:
                    stage_metadata['label_budget'][index] = total_avaialble_labels
                total_avaialble_labels -= stage_metadata['label_budget'][index]

            return stage_metadata['label_budget']

        else:
            print("Missing stage metadata for", stage)
            exit(1)

    def start_checkpoint(self, stage_name, target_dataset, checkpoint_num):
        """ Cycle through the checkpoints for all stages in order.
        Report an error if we try to start a checkpoint after the last
        stage is complete. A checkpoint is ended when post_results is called.

        Args:
            stage_name (str): name of current stage
            target_dataset (framework.dataset): framework dataset class

        Returns:

        """

        if not self.stagenames:
            print("Can't start a checkpoint without initializing a sesssion")
            exit(1)

        if not self.current_stage == stage_name:
            # this is a new stage, so reset to use the budgets for the new stage
            self.current_stage = stage_name

        # move to the next checkpoint and move its budget into the current budget.
        stage_metadata = self.get_stage_metadata(stage_name)
        self.current_checkpoint_index = checkpoint_num
        if self.current_checkpoint_index >= len(stage_metadata['label_budget']):
            print("Out of checkpoints, cant start a new checkpoint")
            exit(1)

        self.current_budget = stage_metadata['label_budget'][
            self.current_checkpoint_index]
        if target_dataset.unlabeled_size < self.current_budget:
            self.current_budget = target_dataset.unlabeled_size

    def get_remaining_budget(self):
        """
        Return the amount of budget remaining on the current checkpoint.
        The budget reflects and extra labels that have already been used, but is
        not affected by seed labels. The budget indicates the total number of
        files that are permitted to be labeled, and the actual number of labels
        may be higher if there are multiple labels per file.

        Returns:
                The current number of files that can be labeled. Multiple labels may
                be returned per file requested.

        """
        if not self.current_budget is None:
            return self.current_budget
        else:
            print("Must start a checkpoint before requesting a budget")
            exit(1)

    def download_dataset(self, dataset_name, dataset_path):
        """ Downloads the data using torchvision

        Args:
            dataset_name (str): name of dataset to download
            dataset_path (Path): path to folder to download into

        """
        import torchvision

        if dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST(dataset_path,
                                                 train=True,
                                                 transform=None,
                                                 target_transform=None,
                                                 download=True)
            self.create_dataset(dataset, dataset_name, dataset_path)

            dataset = torchvision.datasets.MNIST(dataset_path,
                                                 train=False,
                                                 transform=None,
                                                 target_transform=None,
                                                 download=True)

            self.create_dataset(dataset, dataset_name, dataset_path)

    @staticmethod
    def create_dataset(dataset, dataset_name, dataset_path):
        """

        Args:
            dataset (torchvision.datasets.mnist):
            dataset_name (str): name of dataset
            dataset_path (Path): path to root dataset folder

        """
        dataset_coco = dict()
        dataset_root = dataset_path / 'images'
        dataset_coco['root'] = str(dataset_root)
        dataset_coco['classes'] = dataset.classes
        dataset_coco['class_to_idx'] = dataset.class_to_idx
        # using prefix to seperate out the images
        split = 'test'
        if dataset.train:
            split = 'train'
        images = []

        for itx, data in enumerate(dataset):
            img, label = data
            img_name = dataset_root / f'{split}_{itx}.png'
            img.save(img_name)

            # Add record for coco
            record = dict()
            record['file_name'] = str(img_name)
            record['height'] = img.height
            record['width'] = img.width
            record['image_id'] = itx
            record['category_id'] = label
            images.append(record)

        dataset_coco['images'] = images
        coco_filename = dataset_path / f'{dataset_name}_{split}.coco'
        json_obj = json.dumps(dataset_coco, indent=4)
        with open(coco_filename, "w") as json_file:
            json_file.write(json_obj)

    def get_dataset(self, stage_name, dataset_split, categories=None):
        """ lookup the path to the dataset in the configuration information
        and use that path to construct and return the correct dataset object

        Args:
            stage_name (str): stage you are in for the problem
            dataset_split (str): name of the dataset that you want load
            categories (list[str]): list of the category names in the specific order
                from the target_dataset

        """
        stage_metadata = self.get_stage_metadata(stage_name)
        dataset_name = stage_metadata["datasets"][dataset_split]
        if stage_metadata:
            dataset_path = (Path( self.metadata['development_dataset_location'] /
                            dataset_name) )
        else:
            print("Missing stage metadata for", stage_name)
            exit(1)

        if not (dataset_path / f'{dataset_name}.coco').exists():
            print("Downloading the Dataset")
            (dataset_path / 'images').mkdir(parents=True, exist_ok=True)
            self.download_dataset(dataset_name, dataset_path)

        labels = pd.read_feather(dataset_path / 'labels' / 'labels.feather')
        name = dataset_name + '_' + dataset_split
        # translate the labels provided by pandas into a dict keyed on the filename
        # while we are at it, build the seed labels as well.
        self.label_sets[name] = dict()
        self.label_sets_pd[name] = dict()
        classes = []

        # select one label for each class, using the first label we find for that
        # class.
        # TODO: make the method of determining which labels are seed labels
        #  configurable.
        self.seed_labels[name] = labels.drop_duplicates(subset='class')
        self.label_sets_pd[dataset_name] = labels

        for label in labels.values:
            # add every label to the self.label_sets for this dataset
            # this is the labels that are searchable by filename
            # this will be needed by get_more_labels later.
            self.label_sets[dataset_name][label[1]] = label[0]

        if self.metadata['problem_type'] == "image_classification":
            return ImageClassificationDataset(self,
                                              dataset_root=dataset_path,
                                              dataset_name=name,
                                              categories=categories)
        else:
            return ObjectDetectionDataset(self,
                                          dataset_root=dataset_path,
                                          dataset_name=name,
                                          categories=categories)

    def get_dataset_jpl(self, stage_name, dataset_split, categories=None):
        """
        lookup the path to the dataset in the configuration information
        and use that path to construct and return the correct dataset object

        Args:
            stage_name:
            dataset_split:
            categories:

        Returns:

        """

        # TODO: Get this working for the local example
        stage_metadata = self.get_stage_metadata(stage_name)
        current_dataset = stage_metadata["datasets"][dataset_split]
        if dataset_split == 'eval':
            dataset_split = 'test'

        dataset_root = (f'{self.metadata["external_dataset_location"]}/{current_dataset}/'
                        f'{current_dataset}_full/{dataset_split}')
        label_file = (f'{self.metadata["external_dataset_location"]}/{current_dataset}/'
                      f'labels_full/labels_{dataset_split}.feather')

        labels = pd.read_feather(label_file)
        name = current_dataset + '_' + dataset_split
        # translate the labels provided by pandas into a dict keyed on the filename
        # while we are at it, build the seed labels as well.
        self.label_sets[name] = dict()
        self.label_sets_pd[name] = dict()
        classes = []

        # select one label for each class, using the first label we find for that
        # class.
        # TODO: make the method of determining which labels are seed labels
        #  configurable.
        self.seed_labels[name] = labels.drop_duplicates(subset='class')
        self.label_sets_pd[name] = labels

        for label in labels.values:
            # add every label to the self.label_sets for this dataset
            # this is the labels that are searchable by filename
            # this will be needed by get_more_labels later.
            self.label_sets[name][label[1]] = label[0]

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

    def get_more_labels(self, fnames, dataset_name):
        """
        Request labels for a given set of filenames. This will deduct the
        number of files requested from the budget, and return all labels for
        the files requested. This can return more labels than files requested
        if individual files have multiple labels per file.
        If not enough budget is available for the requested list, this function
        will produce an error.

        Args:
            fnames (list[str]): name of file names
            dataset_name (str): lookup for dataset name in label_set_pd

        Returns:

        """
        if self.current_budget is None:
            print("Can't get labels before checkpoint is started")
            exit(1)

        if len(fnames) > self.current_budget:
            # the request is for too many labels
            print("Too many labels requested, not enough budget.")
            exit(1)

        mask =  self.label_sets_pd[dataset_name]['id'].isin(fnames)
        new_labels = self.label_sets_pd[dataset_name][mask]
        return new_labels.to_dict()

    def get_seed_labels(self, dataset_name, seed_level):
        """ seed labels do not count against the budgets

        Args:
            dataset_name (str): name of dataset to get the seed labels from
            seed_level: (int): number of seed label level

        Returns:

        """
        # TODO: get secondary seed labels added here
        return self.seed_labels[dataset_name]

    def post_results(self, stage_id, dataset, predictions):
        """ Submit predictions for analysis and recording. This function
        will report on the accuracy of the submitted predictions.

        Args:
            dataset (framework.dataset):  Framework Dataset
            predictions:

        Returns:

        """
        # TODO: Currently this simply writes the results to the results_file
        # this will need to do more processing in the future.
        self.current_budget = None
        predictions_formatted = dataset.format_predictions(
            predictions[0], predictions[1])

        predicitons_filename = self.metadata['results_file']
        json_obj = json.dumps(predictions_formatted, indent=4)
        with open(predicitons_filename, "w") as json_file:
            json_file.write(json_obj)

        gt = self.label_sets_pd[dataset.name].sort_values('id')
        pred = pd.DataFrame(predictions_formatted).sort_values('id')
        if self.metadata['problem_type'] == 'image_classification':
            # Ensure that this is true and all classes are aligned.
            assert ((gt['id'].values != pred['id'].values).sum() == 0)
            acc2 = (gt['class'].values == pred['class'].values).mean()

            from .metrics import accuracy
            acc = accuracy(pred, gt)
            print(f'Accuracy for Stage:{stage_id} '
                  f'Checkpoint: {self.current_checkpoint_index} is '
                  f'{100 * acc:.02f}%')

        elif self.metadata['problem_type'] == 'object_detection':
            from .metrics import mAP
            acc = mAP(pred, gt)
            print(f'Accuracy for Stage:{stage_id} '
                  f'Checkpoint: {self.current_checkpoint_index} is '
                  f'mAP: {100 * acc:.02f}')

    def get_problem_metadata(self, task_id):
        """
        Return the metadata for the given task id. This data will
        provide information about the test configuration for the task.

        Args:
            task_id:

        Returns:

        """
        self.metadata = self.configuration_data[task_id]
        return self.metadata

    def terminate_session(self):
        """
        End the current session. This should be called after all results have been posted,
        and before a new session is started.
        """
        self.toolset = dict()
        self.metadata = None

    def get_stage_metadata(self, stagename):
        """
        Find and return the metadata for the given stage name.
        If no matching stage is found, this will return None.
        """

        # search through the list of stages for one that has a matching name.
        for stage in self.metadata["stages"]:
            if stage['name'] == stagename:
                return stage
        return None

    def format_status(self, update: bool) -> str:
        """
         Update and return formatted string with the current status
         of the problem/task.  Also should return accuracy if available

         Args:
             update (bool): not used

         Returns:
               str: Formatted String of Status
         """
        info = json.dumps(self.metadata, indent=4)
        return '\n'.join(['Problem/Task Status:', info, ''])
