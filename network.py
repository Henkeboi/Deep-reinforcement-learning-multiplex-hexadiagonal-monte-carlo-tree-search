from hex import Hex
from mct import MCT
import random
import copy
import operator
import torch
import numpy as np
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, hidden_layers, activation_functions, output_size):
        super(Network, self).__init__()
        self.layers = torch.nn.ModuleList()
        layer = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3))
        self.layers.append(layer)
        layer = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2))
        self.layers.append(layer)
        layer = torch.nn.Flatten()
        self.layers.append(layer)
        layer = torch.nn.Linear(288, output_size)
        self.layers.append(layer)
        self.activation_functions = activation_functions

        #for i in range(0, len(hidden_layers) - 1):
        #    input_size = hidden_layers[i]
        #    layer = torch.nn.Linear(input_size, hidden_layers[i + 1])
        #    self.layers.append(layer)

    def forward(self, tensor):
        tensor = tensor.reshape(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])
        for i, layer in enumerate(self.layers):
            if self.activation_functions[i].lower() == 'relu':
                tensor = torch.relu(layer(tensor))
            elif self.activation_functions[i].lower() == 'linear':
                tensor = layer(tensor)
            elif self.activation_functions[i].lower() == 'sigmoid':
                tensor = torch.sigmoid(layer(tensor))
            elif self.activation_functions[i].lower() == 'tanh':
                tensor = torch.tanh(layer(tensor))
        return tensor
