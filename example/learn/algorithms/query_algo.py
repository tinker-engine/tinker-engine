import sys
FROM BASEalgorithm import BaseAlgorithm
import numpy as np


class DomainNetworkSelection(BaseAlgorithm):
    def __init__(self, arguments):
        BaseAlgorithm.__init__(self, arguments)

    def execute(self, toolset, step_descriptor):
        # stage is a string that is passed in acording to the protocol. It identifies which
        # stage of the tets is being requested (e.g. "train", "adapt" )
        # execution does not return anything, it is purely for the sake of altering the internal model.
        # Available reources for training can be retreived using the BaseAlgorithm functions.
        self.toolset = toolset
        if step_descriptor == 'SelectAndLabelData':
            self.select_and_label_data()

        pass

    def select_and_label_data(self):
        # ###################  Train Your Approach ###############
        #  Here you can train your approach.  This approach initializes the
        #  networks for each budget level separately though that isn't required.
        #  You could have persistence though different budget levels if you want.
        #  Everything between here and the active learning is where your code
        #  should implement the learning.
        #
        #   Note: Your code should be able to adapt to having no labeled or
        #       unlabeled data (at least be able to run)

        # Initialize/Re-Initialize models

        #
        # ###################  End of Label Selection Your Approach ###############

        # ##################  ACTIVE LEARNING... Finally. #####################
        #  Figure out the current budget left before checkpoint/evaluation from
        #  the status
        budget = self.toolset['budget']

        #  This approach sets the budget for how many images that they want
        #  labeled.
        # self.vaal.sampler = VAAL.sampler.AdversarySampler(budget)

        #        # You pick which indices that you want labeled.  Here, it is using
        #        # dataset to ensure the correct indices and tracking inside their
        #        # function (the dataset getitem returns the index)
        #         sampled_indices = self.vaal.sample_for_labeling(
        #             vae, discriminator, unlabeled_dataloader)
        #
        #        #  ########### Query for labels -- Kitware managed ################
        #        #  This function is handled by Kitware and takes the indices from the
        #        #  algorithm and queries for new labels. The new labels are added to
        #        #  the dataset and the labeled/unlabeled indices are updated
        sampled_indices = np.random.choice(
            list(self.toolset["target_dataset"].unlabeled_indices),
            budget
        )

        self.toolset["target_dataset"].get_more_labels(sampled_indices)


    #        #  Note: you don't have to request the entire budget, but
    #        #      you shouldn't end the function until the budget is exhausted
    #        #      since the budget is lost after evaluation.

    #
    # ##################  End of ACTIVE LEARNING #####################
