from hex import Hex
from mct import MCT
import random
import copy
import operator
import torch
import numpy as np
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, hidden_layers, output_size):
        super(Network, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.r_buffer = []
        for i in range(0, len(hidden_layers) - 1):
            input_size = hidden_layers[i]
            layer = torch.nn.Linear(input_size, hidden_layers[i + 1])
            self.layers.append(layer)

    def forward(self, tensor):
        output = None
        for i, layer in enumerate(self.layers):
            tensor = torch.relu(layer(tensor))
        output = torch.tanh(tensor)
        return output
