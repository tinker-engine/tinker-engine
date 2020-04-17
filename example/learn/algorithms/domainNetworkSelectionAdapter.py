import abc
from basealgorithm import BaseAlgorithm

class DomainNetworkSelectionAdapter(BaseAlgorithm):
    ''' Adapt the DomainNetworkSelection class for use with the
        framework
    '''

    def __init__(self, arguments):
        BaseAlgorithm.__init__(self, arguments)


    def execute(self, toolset, step_descriptor):
        ''' Redirect the execute call to the correct subfunction name
            and break out the toolset for simpler use

        Arguments:
            toolset: dict containing named resources for the
                functions to use
            step_descriptor: string describing which step of
                processing should be done

        Returns: the information returned by the subfunction
        '''
        self.target_dataset = toolset["target_dataset"]
        self.Whitelist_Datasets = toolset["whitelist_datasets"]
        if step_descriptor == "SelectNetworkAndDataset":
            return self.SelectNetworkAndDataset()
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def SelectNetworkAndDataset(self):
        ''' a subfunction dispatched by execute. This function must
            implement a part of the algorithm itself.
        '''
        raise NotImplementedError

    
