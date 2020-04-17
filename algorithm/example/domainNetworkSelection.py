import sys
import torchvision.models as models
sys.path.append("..")
from basealgorithm import BaseAlgorithm


class DomainNetworkSelection(BaseAlgorithm):
    def __init__(self, arguments):
        BaseAlgorithm.__init__(self, arguments)

    def execute(self, toolset, step_descriptor):
        # stage is a string that is passed in acording to the protocol. It identifies which
        # stage of the task is being requested (e.g. "train", "adapt" )
        # execution does not return anything, it is purely for the sake of altering the internal model.
        # Available reources for training can be retreived using the BaseAlgorithm functions.
        self.toolset = toolset


        if step_descriptor == 'SelectNetworkAndDataset':
            return self.SelectNetworkAndDataset()

    def SelectNetworkAndDataset(self):
        ''' Select the Closest Network and Dataset for the Task

        Returns: Nearest Network, Nearest Dataset

        '''
        pretrained_network = models.resnet18(pretrained=True)
        dataset =  self.toolset['whitelist_datasets']['imagenet_1k_train']
        return pretrained_network, dataset
