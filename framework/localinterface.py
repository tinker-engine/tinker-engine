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
            print(f'Loading {name}')
            for dset in ['train', 'test']:
                labels = pd.read_feather(e / 'labels' / f'labels_{dset}.feather')
                e_root = e / f'{name}' / dset
                if 'bbox' in labels.columns:
                    external_datasets[f'{name}_{dset}'] = ObjectDetectionDataset(self,
                          dataset_root=e_root,
                          dataset_id=f'{name}_{dset}',
                          seed_labels=labels)
                else:
                    external_datasets[f'{name}_{dset}'] = ImageClassificationDataset(self,
                          dataset_root=e_root,
                          dataset_id=f'{name}_{dset}',
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
        #TODO:
        pass

    def update_external_datasets(self):
        target_name = self.toolset["target_dataset"].name
        train_id = f'{target_name}_train'
        test_id = f'{target_name}_test'
        self.toolset["whitelist_datasets"][train_id] = self.toolset["target_dataset"]
        self.toolset["whitelist_datasets"][test_id] = self.toolset["eval_dataset"]

    def get_dataset(self, stage_name, dataset_name categories=None):
        dataset_path = self.metadata["stages"][stage_name]["datasets"][dataset_name]
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
        # TODO:
        pass

    def post_results(self, predictions):
        eval_dataset = self.get_evaluation_dataset()
        predictions = eval_dataset.format_predictions(predictions[0], predictions[1])
        predicitons_filename = self.metadata['results_file']
        json_obj = json.dumps(predictions, indent = 4)
        with open(predicitons_filename, "w") as json_file:
            json_file.write(json_obj)

    def get_problem_metadata(self, task_id=None):
        return self.metadata

    def terminate_session(self):
        self.toolset = dict()
        self.metadata = None

    def get_stages(self):
        return self.metadata["stages"].keys()

