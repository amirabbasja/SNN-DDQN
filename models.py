# A file containing the various network designs to use
import torch.nn as nn
import snntorch as snn
import torch

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

class qNetwork_SNN(nn.Module):
    def __init__(self, layerSizes, **kwargs):
        """
        Initialize the SNN (Spiking Neural Network) with arbitrary number of hidden layers.
        
        Args:
            layers (list): layers is a list of the form: [inputSize, ... Hidden Layers Sizes ... , outputSize]
            **kwargs: Additional parameters including "beta" and "tSteps"
        """
        super().__init__()

        # Impose assetrions
        assert "beta" in kwargs.keys(), "Cna not make a network without beta"
        assert "tSteps" in kwargs.keys(), "Please pass the tSteps parameter"
        assert 2 < len(layerSizes), "Network must have at least, one hidden layer"

        # Model super parameters
        self.beta = kwargs["beta"]
        self.tSteps = kwargs["tSteps"]

        # Build the network layers dynamically
        layers = []
        lifLayers = []
        
        # Create hidden layers and their corresponding LIF neurons
        for i in range(len(layerSizes) - 1):
            # Linear layer
            layer = nn.Linear(layerSizes[i], layerSizes[i + 1])
            layers.append(layer)
            
            # LIF neuron layer
            lifLayer = snn.Leaky(beta=self.beta)
            lifLayers.append(lifLayer)
        
        # Register layers as module lists for proper parameter management
        self.layers = nn.ModuleList(layers)
        self.lifLayers = nn.ModuleList(lifLayers)

    def forward(self, x):
        # Initialize potentials for all layers (hidden + output)
        potentials = []
        for lifLayer in self.lifLayers:
            potentials.append(lifLayer.reset_mem())

        # Save the state of the output layer
        outSpikes = []
        outPotentials = []

        # Iterate through time steps
        for t in range(self.tSteps):
            current_input = x
            
            # Process through all layers
            for i, (layer, lifLayer) in enumerate(zip(self.layers, self.lifLayers)):
                # Apply linear transformation
                current = layer(current_input)
                
                # Apply LIF dynamics
                spike, potentials[i] = lifLayer(current, potentials[i])
                
                # Set input for next layer
                current_input = spike
                
                # If this is the output layer, save the results
                if i == len(self.layers) - 1:  # Last layer (output)
                    outSpikes.append(spike)
                    outPotentials.append(potentials[i])

        return torch.stack(outSpikes, dim=0).sum(dim=0)