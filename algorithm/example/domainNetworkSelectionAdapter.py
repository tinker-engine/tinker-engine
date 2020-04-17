import abc
from basealgorithm import BaseAlgorithm

class DomainNetworkSelectionAdapter(BaseAlgorithm):
    def __init__(self, arguments):
        BaseAlgorithm.__init__(self, arguments)


    def execute(self, toolset, step_descriptor):
        self.target_dataset = toolset["target_dataset"]
        self.Whitelist_Datasets = toolset["Whitelist_Datasets"]
        if step_descriptor == "SelectNetworkAndDataset":
            return self.SelectNetworkAndDataset()
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def SelectNetworkAndDataset(self):
        raise NotImplementedError

    
