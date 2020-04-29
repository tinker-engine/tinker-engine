from framework.basealgorithm import BaseAlgorithm
import numpy as np
import torch

import sampler
import solver
import model


class DomainNetworkSelection(BaseAlgorithm):
    def __init__(self, toolset):
        BaseAlgorithm.__init__(self, toolset)
        self.config = dict()

    def execute(self, toolset, step_descriptor):
        # stage is a string that is passed in according to the protocol. It
        # identifies which stage of the task is being requested (e.g. "adapt",
        # "Evaluate" ) execution does not return anything, it is purely for the sake
        # of altering the internal model. Available resources for training can be
        # retrieved using the BaseAlgorithm functions.
        self.toolset = toolset
        if step_descriptor == 'Initialize':
            self.select_and_label_data()
        if step_descriptor == 'SelectAndLabelData':
            self.select_and_label_data()

        pass

    def initialize(self):
        # ############# Example Specific Attributes ####################
        # Here is where you can add your own attributes for your algorithm

        # set the VAAL configuration options
        self.config["batch_size"] = 128
        self.config["num_workers"] = 8
        self.config["latent_dim"] = 32
        self.config["cuda"] = True
        self.config["train_iterations"] = 25
        self.config["num_vae_steps"] = 2
        self.config["num_adv_steps"] = 1
        self.config["adversary_param"] = 1
        self.config["beta"] = 1


        self.vaal = solver.Solver(self.config, None)
        self.cuda = self.config["cuda"] and torch.cuda.is_available()
        self.train_accuracies = []
        # self.num_classes = self.base_dataset.num_cats
        # model = CloserLookFewShot.solver.get_model(self.arguments["backbone"])
        # self.task_model = CloserLookFewShot.model.BaselineTrain(
        # model, self.num_classes, self.arguments['cuda'])
        # ############## End of Specific Attributes

    def select_and_label_data(self):

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

        # Initialize/Re-Initialize models
        if self.toolset["target_dataset"].unlabeled_size > 0:
            vae = model.VAE(int(self.config["latent_dim"]))
            discriminator = model.Discriminator(int(self.config["latent_dim"]))

            # train the models on the current data

            acc, task_model, vae, discriminator = self.vaal.train(
                labeled_dataloader,
                None,  # Setting task model to None so it doesn't train
                vae,
                discriminator,
                unlabeled_dataloader,
            )

            #
            # ###################  End of Label Selection Your Approach ###############

            # ##################  ACTIVE LEARNING... Finally. #####################
            #  Figure out the current budget left before checkpoint/evaluation from
            #  the status
            budget = self.toolset['budget']


            #  This approach sets the budget for how many images that they want
            #  labeled.
            self.vaal.sampler = sampler.AdversarySampler(budget)

            # You pick which indices that you want labeled.  Here, it is using
            # dataset to ensure the correct indices and tracking inside their
            # function (the dataset getitem returns the index)
            sampled_indices = self.vaal.sample_for_labeling(
                vae, discriminator, unlabeled_dataloader)

            #  ########### Query for labels -- Kitware managed ################
            #  This function is handled by Kitware and takes the indices from the
            #  algorithm and queries for new labels. The new labels are added to
            #  the dataset and the labeled/unlabeled indices are updated

            self.toolset["target_dataset"].get_more_labels(sampled_indices)


    #        #  Note: you don't have to request the entire budget, but
    #        #      you shouldn't end the function until the budget is exhausted
    #        #      since the budget is lost after evaluation.

    #
    # ##################  End of ACTIVE LEARNING #####################
