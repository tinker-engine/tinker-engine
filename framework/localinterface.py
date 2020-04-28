import abc
import os
import sys
import requests
import json
import inspect
from pathlib import Path
import pandas as pd
from framework.main import protocol_file_path
from framework.dataset import ImageClassificationDataset
from framework.dataset import ObjectDetectionDataset



class LocalInterface:
    def __init__(self, json_configuration_file):

        json_full_path = os.path.join(protocol_file_path, json_configuration_file)
        print( "Protocol path", json_full_path )
        if not os.path.exists(json_full_path):
            print("Given LocalInterface configuration file does not exist")
            exit(1)

        with open(json_full_path) as json_file:
            self.configuration_data = json.load(json_file)
        self.metadata = None
        self.toolset = dict()
        self.label_sets = {}

    def get_task_ids(self):
        # each top level in the configuration_data represents a single
        # task. The keys of this dict are the names of the tasks.
        return self.configuration_data.keys()

    def initialize_session(self, task_id):
        # clear any old session data, and prepare for the next task
        self.metadata = self.configuration_data[task_id]
        self.stagenames = self.get_stages()
        self.current_stage = None


    def get_whitelist_datasets(self):
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
                 external_datasets[name] = ObjectDetectionDataset(self,
                         dataset_root=e,
                         seed_labels=labels)
            else:
                external_datasets[name] = ImageClassificationDataset(self,
                        dataset_root=e,
                        seed_labels=labels)

        return external_datasets


    def get_budget_checkpoints(self, stage):
        """
        Find and return the budget checkpoints from the previously loaded metadata
        """
        stage_metadata = self.get_stage_metadata(stage)
        if stage_metadata:
            return stage_metadata['label_budget']
        else:
            print( "Missing stage metadata for", stage )
            exit(1)

    def start_next_checkpoint(self, stage_name):
        # cycle thorugh the checkpoints for all stages in order.
        # report an error if we try to start a checkpoint after the last
        # stage is complete. A checkpoint is ended when post_results is callled.

        if not self.stagenames:
            print("Can't start a checkpoint without initializing a sesssion")
            exit(1)

        if not self.current_stage == stage_name:
            # this is a new stage, so reset to use the budgets for the new stage
            self.current_checkpoint_index = -1 #this will increment to be the 0th item automatically
            self.current_stage = stage_name

        # move to the next checkpoint and moive its budget into the current budget.
        stage_metadata = self.get_stage_metadata(stage)
        self.current_checkpoint_index += 1
        if self.current_checkpoint_index >= len( stage_metadata['label_budget'] ):
            print("Out of checkpoints, cant start a new checkpoint")
            exit(1)

        self.current_budget = stage_metadata['label_budget'][self.current_checkpoint_index]

    def get_remaining_budget(self):
        if self.current_budget:
            return self.current_buget
        else:
            print("Must start a checkpoint before requesting a budget")
            exit(1)


    def update_external_datasets(self):
        target_name = self.toolset["target_dataset"].name
        train_id = f'{target_name}_train'
        test_id = f'{target_name}_test'
        self.toolset["whitelist_datasets"][train_id] = self.toolset["target_dataset"]
        self.toolset["whitelist_datasets"][test_id] = self.toolset["eval_dataset"]

    def get_dataset(self, stage_name, dataset_name, categories=None):
        # lookup the path to the dataset in the configuration infromation
        # and use that path to construct and return the correct dataset object
        stage_metadata = self.get_stage_metadata( stage_name )
        if stage_metadata:
            dataset_path = stage_metadata["datasets"][dataset_name]
        else:
            print( "Missing stage metadata for", stage_name )
            exit(1)

        e = Path(dataset_path)
        labels = pd.read_feather(e / 'labels' / 'labels.feather')
        self.label_sets[dataset_path] = labels
        if 'bbox' in labels.columns:
            return ObjectDetectionDataset(self,
                    dataset_root=dataset_path,
                    categories=categories)
        else:
            return ImageClassificationDataset(self,
                    dataset_root=dataset_path,
                    categories=categories)

    def get_more_labels(self, fnames):
        if not self.current_budget:
            print("Cen't get labels before checkpoint is started")
            exit(1)
        # TODO: implement
        print("get_more_labels")
        exit(0)

    def get_seed_labels(self, dataset_root):
        # seed labels do not count against the budgets
        print("get_seed_labels")

        # TODO: pare this down to a smaller list.
        return self.label_sets[dataset_root]

    def post_results(self, predictions):
        #TODO: Currently this simply writes the results to the results_file
        # this will need to do more processing in the future.
        self.current_budget = None
        eval_dataset = self.get_evaluation_dataset()
        predictions = eval_dataset.format_predictions(predictions[0], predictions[1])
        predicitons_filename = self.metadata['results_file']
        json_obj = json.dumps(predictions, indent = 4)
        with open(predicitons_filename, "w") as json_file:
            json_file.write(json_obj)

    def get_problem_metadata(self, task_id):
        self.metadata = self.configuration_data[task_id]
        return self.metadata

    def terminate_session(self):
        self.toolset = dict()
        self.metadata = None

    def get_stages(self):
        stagenames = []
        for stage in self.metadata["stages"]:
            stagenames.append(stage['name'])
        return stagenames

    def get_stage_metadata(self, stagename):
        # search through the list of stages for one that has a matching name.
        for stage in self.metadata["stages"]:
            if stage['name'] == stagename:
                return stage
        return None
