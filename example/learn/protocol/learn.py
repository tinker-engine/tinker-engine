from jplinterface import JPLInterface
from baseprotocol import BaseProtocol

class Learn(JPLInterface, BaseProtocol):
    ''' The protocol class should derrive from two source classes. The first is the BaseProtocol. This
        provides the self.get_algorithm function for locating and creating an algorithm object for you
        given only the filename of the file containing the algorithm. To use the functions of that
        algorithm, call the execute() function. This call performs two critical actions that are not
        available by directly calling the algorithms functions. First, it establishes an indirect
        call path that allows the automated tools to generate and annotate the adapter and template
        files for use when creating brand new algoritms. Second, it allows the deconstruction of the
        toolset dict(), and automatically stores the toolset for local use within the algorithm itself.

        The second class that the protocol class should derive from is the desired back-end data and
        reporting structure. One such back-end is the JPLInterface which provides access to the JPL
        TA-1 server for the larn project (and hopefully asil-on as well). Another useful back-end is
        the LocalInterface which provides similar access as the JPLInterface using local data only.
        The interface mimics the JPLInterface, but works entirely within the framework, and requires
        no external server to function.

    '''
    def __init__(self, algorithmsdirectory):
        BaseProtocol.__init__(self, algorithmsdirectory)
        JPLInterface.__init__(self,
                              apikey="adab5090-39a9-47f3-9fc2-3d65e0fee9a2",
                              url='https://api-dev.lollllz.com/')
                              # url="http://myserviceloadbalancer-679310346.us-east-1.elb.amazonaws.com")

    def run_protocol(self):
        ''' run_protocol is called from the framework and represents that initial exeuction point
            of the protocol.
        '''
        taskIDs = self.get_task_ids()
        import ipdb
        ipdb.set_trace()
        for task in taskIDs:
            if self.get_problem_metadata(task)['problem_type'] == 'machine_translation':
                continue
            self.run_task(task)

    def run_task(self, task_id):

        self.initialize_session(task_id)

        self.toolset["whitelist_datasets"] = self.get_whitelist_datasets()

        domain_select_algo = self.get_algorithm("domainNetworkSelection.py")
        algo_select_algo = self.get_algorithm("algoSelection.py")

        for stage in ['base', 'adapt']:
            self.stage_id = stage
            self.toolset["target_dataset"] = self.get_target_dataset()

            source_network, source_dataset = domain_select_algo.execute(
                self.toolset, "SelectNetworkAndDataset")

            self.toolset["source_dataset"] = source_dataset
            self.toolset["source_network"] = source_network

            query_algo_id, adapt_algo_id = algo_select_algo.execute(
                self.toolset, "SelectAlgorithms")

            query_algo = self.get_algorithm(query_algo_id)
            adapt_algo = self.get_algorithm(adapt_algo_id)
            query_algo.execute(self.toolset, "Initialize")
            adapt_algo.execute(self.toolset, "Initialize")

            checkpoints = self.get_budget_checkpoints()

            for checkpoint in checkpoints:
                self.toolset["budget"] = self.status['budget_left_until_checkpoint']
                # call the queryalgo to update the dataset with new labels
                query_algo.execute(self.toolset, "SelectAndLabelData")

                #call the estimatoralgo to update the model to incorporate the new labels
                adapt_algo.execute(self.toolset, "DomainAdaptTraining")
                self.toolset["eval_dataset"] = self.get_evaluation_dataset()
                results = adapt_algo.execute(self.toolset, "EvaluateOnTestDataSet")
                self.post_results(results)
        print(self)

        self.terminate_session()

