from jplinterface import JPLInterface
from baseprotocol import BaseProtocol

class SAILON(JPLInterface, BaseProtocol)
:
    def __init__(self, algorithmsdirectory):
        BaseProtocol.__init__(self, algorithmsdirectory)
        JPLInterface.__init__(self,
                             apikey="abc12334",
                             url="http://foo.bar/baz")

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
