from hex import Hex
from mct import MCT
from network import Network
import random
import copy
import operator
import torch
import numpy as np
import matplotlib.pyplot as plt

class NeuralActor:
    def __init__(self, hidden_layers, activation_functions, num_max_moves, learning_rate, optimizer):
        torch.manual_seed(42)
        random.seed(10)
        self.num_max_moves = num_max_moves
        self.nn = Network(hidden_layers, activation_functions, num_max_moves)
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss() 
        self.rbuffer = []
        if optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate)
        elif optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        elif optimizer.lower() == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.nn.parameters(), lr=self.learning_rate)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.nn.parameters(), lr=self.learning_rate)
     
    def update_Q(self, training_data):
        loss = 0
        self.rbuffer.append(training_data)
        training_data = self.rbuffer[random.randrange(0, len(self.rbuffer))]
        for i in range(0, len(training_data)):
            state, label = training_data[i]
            state = torch.from_numpy(state).float()
            nn_output = self.nn(state) # Forward pass
            nn_output = nn_output.view(1, self.num_max_moves)
            index = [x / sum(label) for x in label]
            label = torch.from_numpy(np.asarray([index])).type(torch.FloatTensor)
            nn_loss = self.loss_function(nn_output, label)
            nn_loss.backward()
            loss += nn_loss.item()
        loss = loss / len(training_data)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def get_action(self, state_str, rand=False):
        state = np.fromstring(state_str, np.int8) - 48
        state = torch.from_numpy(state).float()
        nn_output = self.nn(state) # Forward pass
        move_index = torch.argmax(nn_output.data)
        while not Hex.is_legal(move_index, state_str):
            if rand == True:
                move_index = torch.tensor(random.randrange(0, self.num_max_moves))
            else:
                nn_output.data[move_index] = -1.0
                move_index = torch.argmax(nn_output.data)
        return move_index.item()

    def store_model(self, name):
        torch.save(self.nn.state_dict(), 'data/' + name + '.pth')

    def load_model(self, name):
        self.nn.load_state_dict(torch.load('data/' + name + '.pth'))
