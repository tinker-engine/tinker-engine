import sys
sys.path.append("..")
from basealgorithm import BaseAlgorithm
 


class Example(BaseAlgorithm): #TODO: change the name of the class.

    def __init__(self, problem, arguments):
        BaseAlgorithm.__init__(self, problem, arguments)
        #TODO: Add your initialization code here (if any)

    def execute(self, toolset, step_descriptor):
        # step_descriptor is a string that is passed in according to the protocol. It identifies which
        # stage of the test is being requested (e.g. "train", "adapt" )
        # the toolset contains all other information / data that this algorithm is permitted to have.
        # toolset is a dictionary of descriptions and items that the description represents.
        # (e.g. toolset["Dataset"] will give the dataset object that is provided for this execution step.
        # execution does not return anything, it is purely for the sake of altering the internal model.
        # Available reources for training can be retreived using the BaseAlgorithm functions.
        #TODO: implement your code training code here
        pass

