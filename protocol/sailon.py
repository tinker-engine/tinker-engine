from protocols import JPLProtocol

class SAILON(JPLProtocol):
    def __init__(self, algorithmsdirectory):
        JPLProtocol.__init__(self, algorithmsdirectory)

    def runProtocol(self):
        testIDs = self.getTestIDs()
        sessionID = self.initializeSession()
        algo = self.getAlgorithm(None)
        for test in test_ids:
            algo.execute(toolset, "Initialize")
            dataset = self.getDataset(test)
            toolset["Dataset"] = dataset
            results = algo.execute(toolset, "FeatureExtractor")
            self.postResults(test, results)

        self.terminateSession()
