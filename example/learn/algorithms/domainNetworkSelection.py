from domainNetworkSelectionAdapter import DomainNetworkSelectionAdapter
import torchvision.models as models

class DomainNetworkSelection(DomainNetworkSelectionAdapter):
    def __init__(self, arguments):
        DomainNetworkSelectionAdapter.__init__(self, arguments)

    def SelectNetworkAndDataset(self):
        ''' Select the Closest Network and Dataset for the Task

        Available Resources:
            self.target_dataset
            self.whitelist_datasets

        Returns: Nearest Network, Nearest Dataset

        '''
        pretrained_network = models.resnet18(pretrained=True)
        dataset =  self.whitelist_datasets['imagenet_1k_train']
        return pretrained_network, dataset
