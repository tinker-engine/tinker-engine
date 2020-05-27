import abc
from framework.basealgorithm import BaseAlgorithm


class ImageClassifierAdapter(BaseAlgorithm):
    def __init__(self, toolset):
        BaseAlgorithm.__init__(self, toolset)

    def execute(self, toolset, step_descriptor):
        """
        stage is a string that is passed in according to the protocol. It
        identifies which stage of the task is being requested (e.g. "train",
        "adapt" ) execution does not return anything, it is purely for the sake of
        altering the internal model. Available resources for training can be
        retrieved using the BaseAlgorithm functions.
        """
        self.toolset = toolset

        if step_descriptor == "Initialize":
            return self.initialize()
        elif step_descriptor == "DomainAdaptTraining":
            return self.domain_adapt_training()
        elif step_descriptor == "EvaluateOnTestDataSet":
            return self.inference()
        else:
            raise NotImplementedError(f"Step {step_descriptor} not implemented")

    @abc.abstractmethod
    def domain_adapt_training(self):
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def inference(self):
        raise NotImplementedError
