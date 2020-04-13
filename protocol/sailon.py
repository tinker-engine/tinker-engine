import JPLProtocol from protocol

class SAILON(Protocol)
   
   def __init__(self):
      pass

   def runProtocol(self):
      testIDs = self.getTestIDs()
      sessionID = self.initializeSession()
      algo = self.getAlgorithm(problem, None)
      for test in test_ids:
         algo.execute(toolset, "Initialize")
         dataset = self.getDataset(test)
         toolset["Dataset"] = dataset
         results = algo.execute(toolset, "FeatureExtractor")
         self.postResults(results)
      
      self.terminateSession()
