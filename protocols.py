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
        #TODO: load the apikey
        self.apikey = "abcd123"
        self.headers = {'user_secret': self.apikey}
        #TODO laod the url
        self.url
        #TODO find the problem_id
        self.problem_id = ""

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

        # load and store the session status and metadata.
        self.status = self.get_current_status()

        self.metadata = self.get_problem_metadata()

    def getWhitelistsets(self, test_id):
        #TODO: get the whitelist datasets for this test id from the JPL server
        print("getWhitelistsets")
        pass


    def getBudgetCheckpoints(self):
        #TODO: return a list of budget amounts (1 entry per checkpoint)
        # this can be loaded from self.metadata
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
        # end the current session with the JPL server
        # no formal communication with the JPL server is needed to end the session.
        # wipe the session id so that it can't be inadvertently used again
        self.sessiontoken = None
        try:
            del self.headers['session_token']
        except KeyError:
            pass
        
        print("terminateSession")
        pass
        

    def get_current_status(self):
        """
        Get the current status of the session.
        This will tell you things like the number of images you can query
        before evaluation, location of the dataset, number of classes, etc.
        An example session: ::

            {
                "active": true,
                "budget_left_until_checkpoint": 3000,
                "current_dataset": {
                    "classes": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9"
                    ],
                    "data_url": "/datasets/lwll_datasets/mnist/mnist_full/train",
                    "dataset_type": "image_classification",
                    "name": "mnist",
                    "number_of_channels": 1,
                    "number_of_classes": 10,
                    "number_of_samples_test": 10000,
                    "number_of_samples_train": 60000,
                    "uid": "mnist"
                },
                "current_label_budget_stages": [
                    3000,
                    6000,
                    8000
                ],
                "date_created": 1580411853000,
                "date_last_interacted": 1580411853000,
                "pair_stage": "base",
                "task_id": "problem_test_image_classification",
                "uid": "iN9xVy67QBt71K9ATOqx",
                "user_name": "Berkely",
                "using_sample_datasets": false
            }

        Returns:
            dict[str, str]: status of problem/task
        """
        r = requests.get(f"{self.url}/session_status", headers=self.headers)
        r.raise_for_status()
        status = r.json()['Session_Status']

        return status

    def get_problem_metadata(self):
        """
        Get the task metadata from JPL's server.
        An example: ::

            {
                "adaptation_can_use_pretrained_model": false,
                "adaptation_dataset": "mnist",
                "adaptation_evaluation_metrics": [
                    "accuracy"
                ],
                "adaptation_label_budget": [
                    1000,
                    2000,
                    3000
                ],
                "base_can_use_pretrained_model": true,
                "base_dataset": "mnist",
                "base_evaluation_metrics": [
                    "accuracy"
                ],
                "base_label_budget": [
                    3000,
                    6000,
                    8000
                ],
                "problem_type": "image_classification",
                "task_id": "problem_test_image_classification"
            }

        """
        r = requests.get(
            f"{self.url}/task_metadata/{self.problem_id}",
            headers=self.headers)
        r.raise_for_status()
        metadata = r.json()
        return metadata


