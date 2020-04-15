from protocols import JPLProtocol

class Learn(JPLProtocol):
   def __init__(self, algorithmsdirectory):
       JPLProtocol.__init__(self, algorithmsdirectory, apikey = "abc1234", url = "https://foo.bar/baz")

   def runProtocol(self):
      testIDs = self.getTestIDs()
      sessionID = self.initializeSession()
      domainalgo = self.getAlgorithm("image_classification.py")
      domainalgo.execute(toolset, "Initialize")
      for test in test_ids:
          toolset["Whitelist"] = self.getWhitelistsets()
          queryalgo, estimatoralgo, network, dataset = domainalgo.execute(toolset, "SelectNetworkAndDataset")
          toolset["Dataset"] = dataset
          toolset["Network"] = network
          queryalgo.execute(toolset, "Initialize")
          estimatoralgo.execute(toolset, "Initialize")
          checkpoints = self.getBudgetCheckpoints()
          for checkpoint in checkpoints:
              toolset["Budget"] = checkpoint
              # call the queryalgo to update the dataset with new labels
              toolset["Dataset"] = queryalgo.execute(toolset, "SelectAndLabelData")
              #call the estimatoralgo to update the model to incorporate the new labels
              estimatoralgo.execute(toolset, "DomainAdaptTraining" )
              toolset["TestDataSet"] = self.getEvaluationDataSet()
              results = estimatoralgo.execute(toolset, "EvaluateOnTestDataSet")
              self.postResults(results)
      self.terminateSession()

