from framework.jplinterface import JPLInterface
from framework.baseprotocol import BaseProtocol


class Learn(JPLInterface, BaseProtocol):
    """
        The protocol class should derive from two source classes. The first is the
        BaseProtocol. This provides the self.get_algorithm function for locating and
        creating an algorithm object for you given only the filename of the file
        containing the algorithm. To use the functions of that algorithm, call the
        execute() function. This call performs two critical actions that are not
        available by directly calling the algorithms functions. First, it establishes
        an indirect call path that allows the automated tools to generate and
        annotate the adapter and template files for use when creating brand new
        algorithms.  Second, it allows the deconstruction of the toolset dict(), and
        automatically stores the toolset for local use within the algorithm itself.

        The second class that the protocol class should derive from is the desired
        back-end data and reporting structure. One such back-end is the
        JPLInterface which provides access to the JPL T&E server for the learn
        project (and hopefully sail-on as well). Another useful back-end is the
        LocalInterface which provides similar access as the JPLInterface using
        local data only. The interface mimics the JPLInterface, but works entirely
        within the framework, and requires no external server to function.

    """
    def __init__(self, algorithmsdirectory):
        BaseProtocol.__init__(self, algorithmsdirectory)
        JPLInterface.__init__(self,
                              apikey="adab5090-39a9-47f3-9fc2-3d65e0fee9a2",
                              url='https://api-dev.lollllz.com/')

        # set the CloserLookFewShot configuration options
        self.CloserLookFewShot_config = dict()
        self.CloserLookFewShot_config["batch_size"] = 128
        self.CloserLookFewShot_config["num_workers"] = 8
        self.CloserLookFewShot_config["cuda"] = True
        self.CloserLookFewShot_config["backbone"] = "Conf4S"
        self.CloserLookFewShot_config["start_epoch"] = 0
        self.CloserLookFewShot_config["end_epoch"] = 25
        self.CloserLookFewShot_config["checkpoint_dir"] = "./checkpoints"

        # set the VAAL configuration options
        self.VAAL_config = dict()
        self.VAAL_config["batch_size"] = 128
        self.VAAL_config["num_workers"] = 8
        self.VAAL_config["latent_dim"] = 32
        self.VAAL_config["cuda"] = True
        self.VAAL_config["train_iterations"] = 25
        self.VAAL_config["num_vae_steps"] = 2
        self.VAAL_config["num_adv_steps"] = 1
        self.VAAL_config["adversary_param"] = 1
        self.VAAL_config["beta"] = 1

    def run_protocol(self):
        """ run_protocol is called from the framework and represents that initial
            execution point of the protocol.
        """
        taskIDs = self.get_task_ids()
        for task in taskIDs:
            if self.get_problem_metadata(task)['problem_type'] \
                    == 'machine_translation':
                continue
            self.run_task(task)

    def run_task(self, task_id):

        self.initialize_session(task_id)

        self.toolset["whitelist_datasets"] = self.get_whitelist_datasets()

        domain_select_algo = self.get_algorithm("domainNetworkSelection.py",
                                                self.toolset)
        algo_select_algo = self.get_algorithm("algoSelection.py",
                                              self.toolset)

        for stage in ['base', 'adapt']:
            self.stage_id = stage
            self.toolset["target_dataset"] = self.get_target_dataset()

            source_network, source_dataset = domain_select_algo.execute(
                self.toolset, "SelectNetworkAndDataset")

            self.toolset["source_dataset"] = source_dataset
            self.toolset["source_network"] = source_network

            query_algo_id, adapt_algo_id = algo_select_algo.execute(
                self.toolset, "SelectAlgorithms")

            # TODO: So how to do it correctly???
            # The following line incorrectly uses the self.CloserLookFewShot_config
            # to demonstrate passing configuration toolsets to get_algorithm()
            # query_algo = self.get_algorithm(query_algo_id,
            #                                 self.CloserLookFewShot_config)
            query_algo = self.get_algorithm(query_algo_id, self.toolset)
            query_algo.execute(self.toolset, "Initialize")

            # check if we are supposed to use the same source for both
            # query and adapt. if not, then init the adapt separately.
            if adapt_algo_id == query_algo_id:
                adapt_algo = query_algo
            else:
                adapt_algo = self.get_algorithm(adapt_algo_id, self.toolset)
                adapt_algo.execute(self.toolset, "Initialize")

            checkpoints = self.get_budget_checkpoints()

            for checkpoint in checkpoints:
                self.toolset["budget"] = self.status['budget_left_until_checkpoint']
                # call the query_algo to update the dataset with new labels
                query_algo.execute(self.toolset, "SelectAndLabelData")

                # call the adapt_algo to update the model to incorporate the
                # new labels
                adapt_algo.execute(self.toolset, "DomainAdaptTraining")
                self.toolset["eval_dataset"] = self.get_evaluation_dataset()
                results = adapt_algo.execute(self.toolset, "EvaluateOnTestDataSet")
                self.post_results(results)
        print(self)

        self.terminate_session()

