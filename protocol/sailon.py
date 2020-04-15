from protocols import JPLProtocol

class SAILON(JPLProtocol):
    def __init__(self, algorithmsdirectory):
        JPLProtocol.__init__(self, algorithmsdirectory, apikey = "abc1234", url = "https://foo.bar/baz")

    def runProtocol(self):
        testIDs = self.getTestIDs()
        sessionID = self.initializeSession()
        algo = self.getAlgorithm("feature_extractor.py")
        for test in test_ids:
            algo.execute(toolset, "Initialize")
            toolset["Dataset"] = self.getEvaluationDataSet()
            results = algo.execute(toolset, "FeatureExtractor")
            self.postResults(results)

        self.terminateSession()
