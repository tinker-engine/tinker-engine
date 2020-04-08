import sys
sys.path.append("..")
from basealgorithm import BaseAlgorithm
 


class Example(BaseAlgorithm):

    def __init__(self, problem, toolset):
        BaseAlgorithm.__init__(self, problem, toolset)
        #TODO: Add your initialization code here

    def execute(self, step_descriptor):
        # stage is a string that is passed in according to the protocol. It identifies which
        # stage of the test is being requested (e.g. "train", "adapt" )
        # execution does not return anything, it is purely for the sake of altering the internal model.
        # Available reources for training can be retreived using the BaseAlgorithm functions.
        #TODO: implement your code training code here
        pass

    def test(self,eval_dataset, step_descriptor):
        #TODO: Implement your inference code here.
        pass

