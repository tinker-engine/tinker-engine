from framework.basealgorithm import BaseAlgorithm


class AlgorithmSelection(BaseAlgorithm):
    def __init__(self, toolset):
        BaseAlgorithm.__init__(self, toolset)

    def execute(self, toolset, step_descriptor):
        # step_descriptor is a string that is passed in acording to the protocol. It identifies which
        # stage of the tets is being requested (e.g. "train", "adapt" )

        if step_descriptor == "SelectAlgorithms":
            if "image_classification" == toolset["target_dataset"].type:
                return "query_algo.py", "image_classification.py"
            elif "object_detection" == toolset["target_dataset"].type:
                return "query_algo.py", "object_detector.py"
            else:
                raise NotImplementedError()
