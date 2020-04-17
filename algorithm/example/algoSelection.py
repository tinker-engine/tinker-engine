import sys
sys.path.append("..")
from basealgorithm import BaseAlgorithm


class AlgorithmSelection(BaseAlgorithm):
    def __init__(self, arguments):
        BaseAlgorithm.__init__(self, arguments)

    def execute(self, toolset, step_descriptor):
        # stage is a string that is passed in acording to the protocol. It identifies which
        # stage of the tets is being requested (e.g. "train", "adapt" )
        # execution does not return anything, it is purely for the sake of altering the internal model.
        # Available reources for training can be retreived using the BaseAlgorithm functions.

        if step_descriptor == 'SelectAlgorithms':
            if 'image_classification' == toolset['target_dataset'].type:
                return 'query_algo.py', 'image_classification.py'
            elif 'object_detection' == toolset['target_dataset'].type:
                return 'query_algo.py', 'object_detection.py'
            else:
                raise NotImplementedError()

