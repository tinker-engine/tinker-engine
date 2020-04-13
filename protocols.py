import abc

class JPLProtocol(metaclass=abc.ABCMeta):

    def __init__(self, algodirectory):
        self.algorithmsbase = algodirectory
        toolset = {}
        #TODO: define how the problem is passed in
        problem = None

    @abc.abstractmethod
    def runProtocol(self):
        raise NotImplementedError


    def getAlgorithm(self, problem, algotype):
        #TODO: load the algorithm from the self.algorithmsbase/algotype
        print("getAlgorithm")
        pass

    def getTestIDs(self):
        #TODO: load the test ids (task ids?) from the JPL server
        print ("getTestIDs")
        pass

    def initializeSession(self):
        #TODO: start a new session with the JPL server
        print("initializeSession")
        pass

    def getWhitelistsets(self, test_id):
        #TODO: get the whitelist datasets for this test id from the JPL server
        print("getWhitelistsets")
        pass


    def getBudgetCheckpoints(self):
        #TODO: return a list of budget amounts (1 entry per checkpoint)
        print("getBudgetCheckpoints")
        pass

    def getEvaluationDataSet(self, test_id):
        #TODO: return the dataset to be used for evaluating the run
        print("getEvaluationDataSet")
        pass

    def postResults(self, test_id, results):
        #TODO: post the results to the JPL server for this dataset
        print("postResults")
        pass

    def terminateSession(self):
        #TODO: end the current session with the JPL server
        print("terminateSession")
        pass
        
