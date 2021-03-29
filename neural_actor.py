from state_manager import StateManager
from mct import MCT
import random
import copy
import operator
import torch
import numpy as np

class Network(torch.nn.Module):
    def __init__(self, hidden_layers, output_size):
        super(Network, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(0, len(hidden_layers) - 1):
            input_size = hidden_layers[i]
            self.layers.append(torch.nn.Linear(input_size, hidden_layers[i + 1]))

    def forward(self, tensor):
        output = None
        for i, layer in enumerate(self.layers):
            tensor = torch.sigmoid(layer(tensor))
        output = torch.nn.functional.softmax(tensor, dim=0)
        return output

class NeuralActor:
    def __init__(self, hidden_layers, num_max_moves, learning_rate):
        torch.manual_seed(42)
        self.num_max_moves = num_max_moves
        self.nn = Network(hidden_layers, num_max_moves)
        self.learning_rate = learning_rate
        #self.loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate)
   
    def update_Q(self, training_data):
        state, label = training_data[0]
        state = torch.from_numpy(state).float()
        label = torch.from_numpy(np.asarray(label)).float()
        label = torch.nn.functional.softmax(label, dim=0).float()
        nn_output = self.nn(state) # Forward pass
        nn_output = nn_output.view(3, 1)
        nn_loss = self.loss_function(nn_output, label)
        nn_loss.backward() 
        self.optimizer.step() 
        self.optimizer.zero_grad()
    

def main():
    max_removable_pieces = 3
    hidden_layers = [3, 4, 3] 
    la = 0.4
    neural_actor = NeuralActor(hidden_layers, max_removable_pieces, la)

    number_of_pieces = 40
    state_manager = StateManager(number_of_pieces, max_removable_pieces)
    num_search_games = 1
    num_simulations = 50
    max_depth = 20
    mct = MCT(state_manager, num_search_games, num_simulations, max_depth)

    mct.play_game()
    training_data = mct.get_training_data()
    neural_actor.update_Q(training_data)

    #mct.play_game()


if __name__ == '__main__':
    main()
