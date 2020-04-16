from protocols import JPLProtocol
import ipdb

class Learn(JPLProtocol):
    def __init__(self, algorithmsdirectory):
        JPLProtocol.__init__(self,
                             algorithmsdirectory,
                             apikey="adab5090-39a9-47f3-9fc2-3d65e0fee9a2",
                             url="http://myserviceloadbalancer-679310346.us-east-1.elb.amazonaws.com")

    def runProtocol(self):
        taskIDs = self.getTaskIDs()
        for task in taskIDs:
            self.runTask("problem_test_image_classification")

    def runTask(self, task_id):

        self.initializeSession(task_id)

        self.toolset["Whitelist_Datasets"] = self.getWhitelistsets()

        domain_select_algo = self.getAlgorithm("domainNetworkSelection.py")
        algo_select_algo = self.getAlgorithm("algoSelection.py")
        for stage in ['base', 'adapt']:
            self.stage_id = stage
            self.toolset["target_dataset"] = self.getTargetDataset()

            source_network, source_dataset = domain_select_algo.execute(
                self.toolset,
                "SelectNetworkAndDataset")

            self.toolset["source_dataset"] = source_dataset
            self.toolset["source_network"] = source_network

            query_algo_id, adapt_algo_id = algo_select_algo.execute(
                self.toolset,
                "SelectAlgorithms")

            query_algo = self.getAlgorithm(query_algo_id)
            adapt_algo = self.getAlgorithm(adapt_algo_id)
            query_algo.execute(self.toolset, "Initialize")
            adapt_algo.execute(self.toolset, "Initialize")

            checkpoints = self.getBudgetCheckpoints()

            for checkpoint in checkpoints:
                self.toolset["Budget"] = checkpoint
                # call the queryalgo to update the dataset with new labels
                self.toolset["Dataset"] = query_algo.execute(self.toolset,
                                                            "SelectAndLabelData")

                #call the estimatoralgo to update the model to incorporate the new labels
                adapt_algo.execute(self.toolset,
                                      "DomainAdaptTraining")
                self.toolset["TestDataSet"] = self.getEvaluationDataSet()
                results = adapt_algo.execute(self.toolset,
                                             "EvaluateOnTestDataSet")
                self.postResults(results)

        self.terminateSession()

