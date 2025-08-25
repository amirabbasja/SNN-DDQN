# A file containing the various network designs to use
import torch.nn as nn

class qNetwork_ANN(nn.Module):
    """
    A classic ANN architecture with linear units and ReLu activations
    """
    def __init__(self, layers: list, **kwargs):
        """
        Initializes the model. 
        
        Args: 
            layers (list): layers is a list of the form: [inputSize, ... Hidden Layers Sizes ... , outputSize]
        """
        super().__init__()
        
        assert 2 < len(layers), "Network must have at least, one hidden layer"
        
        lstLayers = []
        
        for i in range(len(layers) - 1):
            if i != len(layers) - 2:
                lstLayers.append(nn.Linear(layers[i], layers[i+1]))
                lstLayers.append(nn.ReLU())
            else:
                lstLayers.append(nn.Linear(layers[i], layers[i+1]))
                
        self.model = nn.Sequential(*lstLayers)

    def forward(self, x):
        return self.model(x)