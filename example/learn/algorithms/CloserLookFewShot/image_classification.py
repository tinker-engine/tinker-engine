import sys
import torch
from imageClassificationAdapter import ImageClassifierAdapter
import torch.utils.data as data

import model
import solver


class ImageClassifierAlgorithm(ImageClassifierAdapter):

    def __init__(self, toolset):
        # def __init__(self, problem, base_dataset, adapt_dataset, arguments):
        ImageClassifierAdapter.__init__(self, toolset)


        self.config = dict()

        self.num_classes = self.toolset.num_cats
        self.task_model = None

        # ############# Example Specific Attributes ####################
        # Here is where you can add your own attributes for your algorithm
        # self.vaal = VAAL.solver.Solver(arguments, None)
        # self.cuda = arguments["cuda"] and torch.cuda.is_available()
        # self.train_accuracies = []
        # model = CloserLookFewShot.solver.get_model(self.arguments["backbone"])

        # ############## End of Specific Attributes

    def initialize(self):
        # ############## initialize the algorithm *************
        # set the CloserLookFewShot configuration options
        self.config["batch_size"] = 128
        self.config["num_workers"] = 8
        self.config["cuda"] = True
        self.config["backbone"] = "Conf4S"
        self.config["start_epoch"] = 0
        self.config["end_epoch"] = 25
        self.config['num_classes'] = self.toolset["target_dataset"].num_cats
        self.CloserLookFewShot_config["checkpoint_dir"] = "./checkpoints"

        # model = solver.get_model(self.arguments["backbone"])
        model = self.toolset["source_network"]
        # TODO: Generalize the feature extraction network input
        self.task_model = model.BaselineTrain(model, self.num_classes,
                                              self.arguments['cuda'])


    def domain_adapt_training(self):
        # ###################  Creating the Labeled DataLoader ###############
        # Create Dataloaders for labeled data by getting the labeled indices and
        # creating a random sub-sampler.  Then create the dataloader using this
        # sampler
        #
        #  Note: To change the transformer in the JPLDataset dataset, edit
        #      current_dataset.transform.  You can also edit the labels during
        #      loading by editing the  current_dataset.target_transform
        #
        #  Note: Another approach could be to inherit the JPLDataset class and
        #      override the __get_item__ function
        #
        #  Check out dataset.py for more information
        #     Notes: There could be NO unlabeled data and the algorithm has to be
        #         able to handle that case!
        #
        #         The min batch size here is so that the batch size isn't
        #         larger than the labeled/unlabeled dataset which causes the
        #         dataloader to hang

        labeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            self.toolset["target_dataset"].get_labeled_indices()
        )

        labeled_dataloader = torch.utils.data.DataLoader(
            self.toolset["target_dataset"],
            sampler=labeled_sampler,
            batch_size=min(self.toolset["target_dataset"].labeled_size,
                           int(self.config["batch_size"])
                           ),
            num_workers=int(self.num_workers),
            collate_fn=self.toolset["target_dataset"].collate_batch,
            drop_last=True,
        )

        # ###################  Creating the Unlabeled DataLoader ###############
        #  Same as labeled dataaset but for unlabeled indices
        unlabeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(
            self.toolset["target_dataset"].get_unlabeled_indices()
        )
        unlabeled_dataloader = torch.utils.data.DataLoader(
            self.toolset["target_dataset"],
            sampler=unlabeled_sampler,
            batch_size=min(self.toolset["target_dataset"].unlabeled_size,
                           int(self.config["batch_size"])
                           ),
            num_workers=int(self.num_workers),
            drop_last=False,
        )

        acc, self.task_model = solver.train(self.config,
                                            labeled_dataloader,
                                            self.task_model)

    def inference(self):
        """
        Inference is during the evaluation stage.  For this example, the
        task_network is trained in the train and adapt stage's code and
        is used here to create the predictions.  The indices are used to
        track the images/filenames for submitting the predictions back to
        JPL

        Args:
            eval_dataset (JPLEvalDataset): Dataset for the labels used
                for evaluation

        Returns:
            tuple(list[int],list[int]): preds and indices
                predicted category indices and image indices

        """
        self.task_model.eval()
        eval_dataloader = data.DataLoader(
            self.toolset["eval_dataset"],
            batch_size=int(self.config["batch_size"]),
            drop_last=False
        )

        preds = []
        indices = []

        for imgs, inds in eval_dataloader:
            if self.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds_ = self.task_model(imgs)

            preds += torch.argmax(preds_, dim=1).cpu().numpy().tolist()
            indices += inds.numpy().tolist()

        preds, indices = self.toolset["eval_dataset"].dummy_data(
                                                       'image_classification')

        return preds, indices



