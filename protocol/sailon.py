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
        sessionID = self.initialize_session()
        algo = self.get_algorithm("feature_extractor.py")
        for test in test_ids:
            algo.execute(toolset, "Initialize")
            toolset["Dataset"] = self.get_evaluation_dataset()
            results = algo.execute(toolset, "FeatureExtractor")
            self.post_results(results)

        self.terminate_session()
