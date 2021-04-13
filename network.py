from state_manager import StateManager
from mct import MCT
import random
import copy
import operator
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import models

class Network(torch.nn.Module):
    def __init__(self, conv_layers, dense_layers, output_size):
        #super(Network, self).__init__()
        #self.output_size = output_size
        #self.layers = torch.nn.ModuleList()
        #last_channel_out = 2
        #for i in range(len(conv_layers)):
        #    f = conv_layers[0][i]
        #    kernel = conv_layers[1][i]
        #    layer = torch.nn.Conv2d(in_channels=last_channel_out, out_channels=f, kernel_size=kernel)
        #    self.layers.append(layer)
        #    last_channel_out = f
        #self.layers.append(torch.nn.Flatten())
        #for i in range(0, len(dense_layers) - 1):
        #    input_size = dense_layers[i]
        #    layer = torch.nn.Linear(input_size, dense_layers[i + 1])
        #    self.layers.append(layer)

        super(Network, self).__init__()
        self.output_size = output_size
        self.layers = torch.nn.ModuleList()
        layer = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(2, 2))
        self.layers.append(layer)
        layer = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2))
        self.layers.append(layer)
        self.layers.append(torch.nn.Flatten())
        for i in range(0, len(dense_layers) - 1):
            input_size = dense_layers[i]
            layer = torch.nn.Linear(input_size, dense_layers[i + 1])
            self.layers.append(layer)

    def forward(self, tensor):
        board = torch.FloatTensor(tensor[0:self.output_size])
        player = [tensor[-1] for i in range(self.output_size)]
        player = torch.FloatTensor(player)
        player = torch.reshape(player, (int(self.output_size ** 0.5), int(self.output_size ** 0.5)))
        board = torch.reshape(board, (int(self.output_size ** 0.5), int(self.output_size ** 0.5)))

        tensor = torch.stack([board, player])
        tensor = torch.reshape(tensor, (1, 2, int(self.output_size ** 0.5), int(self.output_size ** 0.5)))
        output = None
        for i, layer in enumerate(self.layers):
            tensor = torch.relu(layer(tensor))
        output = torch.tanh(tensor)
        return output
