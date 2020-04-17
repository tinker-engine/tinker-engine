import abc
import os
import sys
import requests
import json
import inspect


class LocalInterface:
    def __init__(self, apikey="", url=""):

        # TODO: define the data_type
        self.data_type = "full"

        self.apikey = apikey
        self.headers = {'user_secret': self.apikey}
        self.url = url

        self.task_id = ""
        self.stage_id = ""

    def getTaskIDs(self):
        #TODO:
        pass

    def initializeSession(self, task_id):
        #TODO:
        pass

    def getWhitelistsets(self):
        #TODO:
        pass

    def getBudgetCheckpoints(self):
        #TODO:
        pass

    def getBudgetUntilCheckpoints(self):
        #TODO:
        pass

    def getEvaluationDataSet(self):
        # TODO: return the dataset to be used for evaluating the run
        pass

    def postResults(self, predictions):
        #TODO:
        pass

    def terminateSession(self):
        #TODO:
        pass

