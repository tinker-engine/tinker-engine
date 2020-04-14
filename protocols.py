import abc
import os
import sys
import requests
import json

class JPLProtocol(metaclass=abc.ABCMeta):

    def __init__(self, algodirectory):
        self.algorithmsbase = algodirectory
        toolset = {}
        #TODO: define how the problem is passed in
        self.problem = None
        #TODO: define the data_type
        self.datatype = "full"
        #TODO: laod the apikey
        self.apikey = "abcd123"
        self.headers = {'user_secret': self.apikey}
        #TODO laod the url
        self.url
        #TODO find the problem_id
        self.problem_id = 0

    @abc.abstractmethod
    def runProtocol(self):
        raise NotImplementedError

    def getAlgorithm(self, algotype):
        # load the algorithm from the self.algorithmsbase/algotype

        #validate that the file exists
        algofile = os.path.join(self.algorithmsbase, algotype)
        if not os.path.exists( algofile ):
            print("given algorithm", algotype, "doesnt exist")
            exit(1)

        #TODO: support handling a directory in the future. The idea would be that the directory
        # would contain only the one algorithm file, and that the protocol wouldn't care what the
        # name of the file was.
        if os.path.isdir( algofile ):
            print("algorithm not yet supported on a directory, use a specific file instead")
            raise NotImplementedError

        # get the path to the algo file so that we can append it to the system path
        # this makes the import easier
        argpath, argfile = os.path.split( algofile )
        if argpath:
            sys.path.append(argpath)

        # load the file as a module and create an object of the class type in the file
        # the name of the class doesnt matter, as long as there is only one class in
        # the file.
        argbase, argext = os.path.splitext(argfile)
        if argext == ".py":
            algoimport = __import__(argbase, globals(), locals(), [], 0)
            for name,obj in inspect.getmembers(algoimport):
                if inspect.isclass(obj):
                    foo = inspect.getmodule( obj )
                    if foo == protocolimport:
                        #construct the algorithm object
                        algorithm = obj(self.problem, "")

        else:
            print("Given algorithm is not a python file")
            exit(1)
        
        return algorithm

    def getTestIDs(self):
        #TODO: load the test ids (task ids?) from the JPL server
        print ("getTestIDs")
        pass

    def initializeSession(self):
        """
        Get the session token from JPL's server.  This should only be run once
        per task. The session token defines the instance of the problem being run.

        Returns:
            none

        """
        r = requests.get(
            f"{self.url}/auth/get_session_token/{self.datatype}/{self.problem_id}",
            headers=self.headers)
        r.raise_for_status()

        self.sessiontoken = r.json()['session_token']

        #prep the headers for easier use in the future.
        self.headers['session_token'] = self.sessiontoken

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
        
