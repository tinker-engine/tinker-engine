import abc
import os
import sys
import requests
import json
import inspect
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

    def get_task_ids(self):
        return self.configuration_data.keys()

    def initialize_session(self, task_id):
        self.metadata = self.configuration_data[task_id]

    def get_whitelist_datasets(self):
        from pathlib import Path
        import pandas as pd
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

    def get_budget_until_checkpoints(self):
        #TODO:
        pass

    def update_external_datasets(self):
        target_name = self.toolset["target_dataset"].name
        train_id = f'{target_name}_train'
        test_id = f'{target_name}_test'
        self.toolset["whitelist_datasets"][train_id] = self.toolset["target_dataset"]
        self.toolset["whitelist_datasets"][test_id] = self.toolset["eval_dataset"]

    def get_dataset(self, stage_name, dataset_name, categories=None):
        stage_metadata = self.get_stage_metadata( stage_name )
        if stage_metadata:
            dataset_path = stage_metadata["datasets"][dataset_name]
        else:
            print( "Missing stage metadata for", stage_name )
            exit(1)

        if self.metadata['problem_type'] == "image_classification":
            return ImageClassificationDataset(self,
                    dataset_root=dataset_path,
                    categories=categories)
        else:
            return ObjectDetectionDataset(self,
                    dataset_root=dataset_path,
                    categories=categories)

    def get_more_labels(self, fnames):
        # TODO:
        pass

    def get_seed_labels(self):
        print("get_seed_labels")
        # TODO:
        pass

    def post_results(self, predictions):
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
        for stage in self.metadata["stages"]:
            if stage['name'] == stagename:
                return stage
        return None
