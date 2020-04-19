import sys
from objectDetectorAdapter import ObjectDetectorAdapter

import torch
import torch.utils.data as data
import CloserLookFewShot.model
import CloserLookFewShot.solver
import VAAL.sampler
import VAAL.solver
import VAAL.model
import numpy as np


class ObjectDetectorAlgorithm(ObjectDetectorAdapter):
    """
    Class which runs the algorithm consisting of at least four
    methods: :meth:`__init__`, :meth:`train`, :meth:`adapt`,
    and :meth:`inference`.  You may have more classes if you
    want, but these are the ones called from the :ref:`main.py`
    file.

    This class requires an init, train, adapt, and
    inference methods.  These methods run at the different
    stages of the problem and are called from :ref:`main.py`.
    You can add more attributes or functions as you see fit.

    For example, you want to preserve the train stage task
    model during the adapt stage so you can add in
    ``self.train_task_model`` as an attribute.
    Alternatively, you can just
    save it in a helper class like "solver" here.

    Attributes:
        problem (LwLL): same as input
        base_dataset (JPLDataset): same as input
        current_dataset (JPLDataset): dataset currently in use
            (either the train or adapt stage dataset)
        adapt_dataset (JPLDataset): the adapt stage dataset
            (defined as None until that stage)
        arguments (dict[str, str]): same as input
        cuda (bool): whether or not to use cuda during inference
        vaal (Solver): solver class for VAAL code
        train_accuracies (list): training accuracies
        task_model (torch.nn.Module): pytorch model for bridging
            between train/adapt stages and the eval stage.

    The last four variables (after arguments) are specific to the VAAL algorithm
    so are not necessary for the overall problem.
    """
    def __init__(self, toolset):
        """
        Here is where you can add in your initialization of your algorithm.
        You should also add any variables that need to persist between
        budget levels and/or stages.

        For example, you need to do inference during the
        evaluate stage and thus task_model is saved here for VAAL to predict
        on the evaluation data in :meth:`inference`.

        Args:
            problem (LwLL): a lwll object that contains all the information
                about the problem.  Look at it's definition for more information
            base_dataset (JPLDataset): a JPLDataset which is a pytorch dataset.
                It contains the labeled and unlabeled data for training.
            adapt_dataset (JPLDataset): a JPLDataset which is a pytorch dataset.
                It contains the labeled and unlabeled data for adaption.
            arguments (dict[str, str]): contains all the arguments for the algorithm
                defined in the ``input.json`` file

        """
        ObjectDetectorAdapter.__init__(self, toolset)

        self.batch_size = 32
        self.num_workers = 0


    def initialize(self):
        pass


    def domain_adapt_training(self):
        """ Method for the training in the train stage of the problem.
        Also known as the base stage.

        Here you will train your algorithm and query for new labels until the
        labeling budget is used up. Once the labeling budget is used up and
        your training is finished, return and it will run the inference function.

        You may modify any part of this function however there are some things
        you will need to do here.

        1.  You will need to get the labels from the JPLdataset.  The examples
            are one way to create the pytorch dataloaders (``labeled_dataloader``
            and ``unlabeled_dataloader`` in the code).
        2.  You will need to query for labels.  You can do this by specifying
            the indices of the labels in the :class:`dataset.JPLDataset`.

        In the example in the code, `self.current_dataset` is either train and adapt
        dataset depending on the stage.  For this example, the VAAL approach doesn't
        have a way to adapt from the trianing stage so we can use
        self.current_dataset here.  VAAL is only an active training approach.

        """
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
                           int(self.batch_size)
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
                           int(self.batch_size)
                           ),
            num_workers=int(self.num_workers),
            drop_last=False,
        )
        #
        # ###################  End of the Creating DataLoaders ###############

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
                                           list(self.toolset["Dataset"].unlabeled_indices),
                                           budget
                                           )

        self.toolset["Dataset"].get_more_labels(sampled_indices)
#        #  Note: you don't have to request the entire budget, but
#        #      you shouldn't end the function until the budget is exhausted
#        #      since the budget is lost after evaluation.

        #
        # ##################  End of ACTIVE LEARNING #####################

        # ##################  Training on all labels... #####################

        # Update the labeled sampler and dataloader with the new data.
        # labeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        #     self.current_dataset.get_labeled_indices()
        # )
        #
        # labeled_dataloader = torch.utils.data.DataLoader(
        #     self.current_dataset,
        #     sampler=labeled_sampler,
        #     batch_size=min(len(self.current_dataset.get_labeled_indices()),
        #                    int(self.arguments["batch_size"])
        #                    ),
        #     num_workers=int(self.arguments["num_workers"]),
        #     collate_fn=self.current_dataset.collate_batch,
        #     drop_last=True,
        # )

        # train the models on the current data
        # acc, self.task_model = CloserLookFewShot.solver.train(self.arguments,
        #                                                       labeled_dataloader,
        #                                                       self.task_model)
        # self.train_accuracies.append(acc)

        # ##################  End of ACTIVE LEARNING #####################

        #  Note: Evaluation/inference will happen after this function is over so you
        #      will probably want to continue to train for the task after the active
        #      learning part is done since you just got new labels.

        #  This function should end when the budget is exhausted and your algorithm
        #  is fully trained on the current set of labels. Feel free to turn this
        #  function into a loop containing both the training and active learning
        #  elements if you want to request labels in smaller increments of data
        #  rather than requesting the entire budget here

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
        # self.task_model.eval()
        # eval_dataloader = data.DataLoader(
        #     eval_dataset,
        #     batch_size=int(self.arguments["batch_size"]),
        #     drop_last=False
        # )
        # preds = []
        # indices = []
        #
        # for imgs, inds in eval_dataloader:
        #     if self.cuda:
        #         imgs = imgs.cuda()
        #
        #     with torch.no_grad():
        #         preds_ = self.task_model(imgs)
        #
        #     preds += torch.argmax(preds_, dim=1).cpu().numpy().tolist()
        #     indices += inds.numpy().tolist()

        preds, indices = toolset["eval_dataset"].dummy_data('object_detection')
        return preds, indices
