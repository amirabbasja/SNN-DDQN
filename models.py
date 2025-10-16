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
        # Added None for consistency with SNN model
        return self.model(x), None

class qNetwork_SNN(nn.Module):
    def __init__(self, layerSizes, **kwargs):
        """
        Initialize the SNN (Spiking Neural Network) with arbitrary number of hidden layers.
        
        Args:
            layers (list): layers is a list of the form: [inputSize, ... Hidden Layers Sizes ... , outputSize]
            **kwargs: Can contain following values
                "beta" and "tSteps": Required for defining the spiking neuron structure
                positiveInitWeights (bool): If True, all initial weights of the network will be positive values, 
                avoiding generation of negative potentials in neurons in the earlier stages of training.
                noOutSpikes (bool): If True, the output will not have spike neurons
        """
        super().__init__()

        # Impose assetrions
        assert "beta" in kwargs.keys(), "Cna not make a network without beta"
        assert "tSteps" in kwargs.keys(), "Please pass the tSteps parameter"
        assert 2 < len(layerSizes), "Network must have at least, one hidden layer"

        # Model super parameters
        self.beta = kwargs["beta"]
        self.tSteps = kwargs["tSteps"]
        if "DEBUG" in kwargs.keys():
            self.DEBUG = kwargs["DEBUG"]
        else:
            self.DEBUG = False

        # New flag for positive-only initialization
        self.positiveInitWeights = kwargs.get("positiveInitWeights", False)

        # For saving information. parameters will be not-None only if the DEBUG flag is True
        self.info = {
            "totalSpikes": None,
            "spikesPerLayer": [None for _ in range(len(layerSizes))]
        } 

        # Build the network layers dynamically
        layers = []
        lifLayers = []
        
        # Layers and their corresponding LIF neurons
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

        # Apply positive-only initialization if requested
        if self.positiveInitWeights:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.abs_()
                    if layer.bias is not None:
                        layer.bias.data.abs_()

    def forward(self, x):
        # Initialize potentials for all layers (hidden + output)
        potentials = []
        for lifLayer in self.lifLayers:
            potentials.append(lifLayer.reset_mem())

        # Save the state of the output layer
        outSpikes = []
        outPotentials = []

        # Iterate through time steps
        _totalSpikes = 0
        _spikesPerLayer = [0 for _ in range(len(self.layers))]

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
                
                # Save info
                if self.DEBUG:
                    _spikesPerLayer[i] += spike.sum().item()
                    _totalSpikes += spike.sum().item()
                    self.info["spikesPerLayer"][i] = _spikesPerLayer[i]
                    self.info["totalSpikes"] = _totalSpikes
                
                # If this is the output layer, save the results
                if i == len(self.layers) - 1:  # Last layer (output)
                    outSpikes.append(spike)
                    outPotentials.append(potentials[i])
        return torch.stack(outSpikes, dim=0).sum(dim=0), self.info

# Utility functions for models

def computeGradientNorms(model):
    """
    Computes the L2 norm for molde layer gradients. 

    Args:
        model (torch.nn.Module): The model to compute the gradient norms for.
    
    Returns:
        L2 norm of the entire model, layer-wise L2 norms
    """
    totNorm = 0.0
    layerNorms = {}
    
    for name, p in model.named_parameters():
        if p.grad is not None:  # Check if gradients exist
            # Compute L2 norm for the layer
            paramNorm = p.grad.data.norm(2).item()
            layerNorms[name] = paramNorm
            # Add to total norm (sum of squares)
            totNorm += paramNorm ** 2
    
    # Finalize total norm by taking square root
    totNorm = totNorm ** 0.5
    
    return totNorm, layerNorms