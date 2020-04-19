import abc
from framework.basealgorithm import BaseAlgorithm

class ObjectDetectorAdapter(BaseAlgorithm):

    def __init__(self, toolset):
        BaseAlgorithm.__init__(self, toolset)

    def execute(self, toolset, step_descriptor):

        self.toolset = toolset
        if step_descriptor == 'Initialize':
            return self.initialize()
            pass
        elif step_descriptor == 'DomainAdaptTraining':
            return self.domain_adapt_training()
        elif step_descriptor == 'EvaluateOnTestDataSet':
            return self.inference()
        else:
            raise NotImplementedError(f'Step {step_descriptor} not implemented')

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def domain_adapt_training(self):
        raise NotImplementedError


    @abc.abstractmethod
    def inference(self):
        raise NotImplementedError
