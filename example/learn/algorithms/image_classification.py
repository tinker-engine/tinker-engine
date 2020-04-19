import sys
import torch
from imageClassificationAdapter import ImageClassifierAdapter
import torch.utils.data as data
# import CloserLookFewShot.model
# import CloserLookFewShot.solver
# import VAAL.sampler
# import VAAL.solver
# import VAAL.model


class ImageClassifierAlgorithm(ImageClassifierAdapter):

    def __init__(self, toolset):
        #def __init__(self, problem, base_dataset, adapt_dataset, arguments):
        ImageClassifierAdapter.__init__(self, toolset)

        self.batch_size = 32
        self.num_workers = 0
        # ############# Example Specific Attributes ####################
        # Here is where you can add your own attributes for your algorithm
        # self.vaal = VAAL.solver.Solver(arguments, None)
        # self.cuda = arguments["cuda"] and torch.cuda.is_available()
        # self.train_accuracies = []
        # self.num_classes = self.base_dataset.num_cats
        # model = CloserLookFewShot.solver.get_model(self.arguments["backbone"])
        # self.task_model = CloserLookFewShot.model.BaselineTrain(
        # model, self.num_classes, self.arguments['cuda'])
        # ############## End of Specific Attributes

    def initialize(self):
        # ############## initialize the algorithm *************
        pass


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

        # ###################  Find Labels for Active Learning ###############
        #  Here you can run active learning.  This approach initializes the
        #  networks for each budget level separately though that isn't required.
        #  You could have persistence though different budget levels if you want.
        #  Everything between here and the active learning is where your code
        #  should implement the learning.
        #
        #   Note: Your code should be able to adapt to having no labeled or
        #       unlabeled data (at least be able to run)

        # Initialize/Re-Initialize models
        # if toolset["Dataset"].unlabeled_size > 0:
        #     vae = VAAL.model.VAE(int(self.arguments["latent_dim"]))
        #     discriminator = VAAL.model.Discriminator(int(self.arguments["latent_dim"]))
        #
        #     # train the models on the current data
        #
        #     acc, task_model, vae, discriminator = self.vaal.train(
        #         labeled_dataloader,
        #         None,  # Setting task model to None so it doesn't train
        #         vae,
        #         discriminator,
        #         unlabeled_dataloader,
        #     )
        #
        #     #
        #     # ###################  End of Label Selection Your Approach ###############
        #
        #     # ##################  ACTIVE LEARNING... Finally. #####################
        #     #  Figure out the current budget left before checkpoint/evaluation from
        #     #  the status
        #     budget = self.problem.get_current_status['budget_left_until_checkpoint']
        #
        #     #  This approach sets the budget for how many images that they want
        #     #  labeled.
        #     self.vaal.sampler = VAAL.sampler.AdversarySampler(budget)
        #
        #      # You pick which indices that you want labeled.  Here, it is using
        #      # dataset to ensure the correct indices and tracking inside their
        #      # function (the dataset getitem returns the index)
        #     sampled_indices = self.vaal.sample_for_labeling(
        #         vae, discriminator, unlabeled_dataloader)
        #
        #      #  ########### Query for labels -- Kitware managed ################
        #      #  This function is handled by Kitware and takes the indices from the
        #      #  algorithm and queries for new labels. The new labels are added to
        #      #  the dataset and the labeled/unlabeled indices are updated
        #     toolset["Dataset"].get_more_labels(sampled_indices)
        #      #  Note: you don't have to request the entire budget, but
        #      #      you shouldn't end the function until the budget is exhausted
        #      #      since the budget is lost after evaluation.
        #
        #     #
        #     # ##################  End of ACTIVE LEARNING #####################
        #
        # # ##################  Training on all labels... #####################
        #
        # # Update the labeled sampler and dataloader with the new data.
        # labeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        #     toolset["Dataset"].get_labeled_indices()
        # )
        #
        # labeled_dataloader = torch.utils.data.DataLoader(
        #     toolset["Dataset"],
        #     sampler=labeled_sampler,
        #     batch_size=min(toolset["Dataset"].labeled_size,
        #                    int(self.arguments["batch_size"])
        #                    ),
        #     num_workers=int(self.arguments["num_workers"]),
        #     collate_fn=toolset["Dataset"].collate_batch,
        #     drop_last=True,
        # )
        #
        # # train the models on the current data
        # acc, self.task_model = CloserLookFewShot.solver.train(self.arguments,
        #                                                       labeled_dataloader,
        #                                                       self.task_model)
        # self.train_accuracies.append(acc)

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
        #     toolset["Dataset"],
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

        preds, indices = self.toolset["eval_dataset"].dummy_data('image_classification')
        return preds, indices



