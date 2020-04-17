from jplinterface import JPLInterface
from baseprotocol import BaseProtocol
import ipdb

class Learn(JPLInterface, BaseProtocol):
    def __init__(self, algorithmsdirectory):
        BaseProtocol.__init__(self, algorithmsdirectory)
        JPLInterface.__init__(self,
                              apikey="adab5090-39a9-47f3-9fc2-3d65e0fee9a2",
                              url='https://api-dev.lollllz.com/')
                              # url="http://myserviceloadbalancer-679310346.us-east-1.elb.amazonaws.com")

    def run_protocol(self):
        taskIDs = self.get_task_ids()
        for task in taskIDs:
            self.run_task("problem_test_image_classification")

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

        self.terminate_session()

