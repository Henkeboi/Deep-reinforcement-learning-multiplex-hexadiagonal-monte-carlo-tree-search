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
            tensor = torch.relu(layer(tensor))
        output = torch.nn.functional.softmax(tensor, dim=0)
        return output

class NeuralActor:
    def __init__(self, hidden_layers, num_max_moves, learning_rate):
        torch.manual_seed(42)
        self.num_max_moves = num_max_moves
        self.nn = Network(hidden_layers, num_max_moves)
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate)
   
    def update_Q(self, training_data):
        for i in range(0, len(training_data)):
            state, label = training_data[i]
            state = torch.from_numpy(state).float()
            nn_output = self.nn(state) # Forward pass
            nn_output = nn_output.view(1, self.num_max_moves)
            max_val = max(label)
            index = list.index(label, max_val)
            label = torch.from_numpy(np.asarray([index]))
            nn_loss = self.loss_function(nn_output, label)
            nn_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, state):
        state = np.fromstring(state, dtype=int, sep=".")
        state = torch.from_numpy(state).float()
        nn_output = self.nn(state) # Forward pass
        move_index = torch.argmax(nn_output.data)
        print(nn_output.data)

        while move_index > state[0] - 1:
            move_index = move_index - 1 # Choose closest move

        return move_index

def main():
    max_removable_pieces = 3
    state_space_variables = 3
    hidden_layers = [state_space_variables, 10, max_removable_pieces] 
    la = 0.1
    number_of_pieces = 8
    state_manager = StateManager(number_of_pieces, max_removable_pieces)
    num_search_games = 1
    num_simulations = 20
    max_depth = 20
    player1 = NeuralActor(hidden_layers, max_removable_pieces, la)
    player2 = NeuralActor(hidden_layers, max_removable_pieces, la)
    mct1 = MCT(player1, num_search_games, num_simulations, max_depth)
    mct2 = MCT(player2, num_search_games, num_simulations, max_depth)

    for i in range(0, 10000):
        mct1.play_game(copy.deepcopy(state_manager))
        training_data = mct1.get_training_data()
        player1.update_Q(training_data)

    for i in range(0, 10000):
        mct2.play_game(copy.deepcopy(state_manager))
        training_data = mct2.get_training_data()
        player2.update_Q(training_data)

    while not state_manager.is_finished():
        if state_manager.player1_to_move():
            print("Player 1 moving")
            move = state_manager.get_moves()[player1.get_action(state_manager.string_representation())]
        elif state_manager.player2_to_move():
            print("Player 2 moving")
            move = state_manager.get_moves()[player2.get_action(state_manager.string_representation())]
        state_manager.make_move(move)
        print(move)
        print()

    if state_manager.player1_won():
        print("Player1 won")
    elif state_manager.player2_won():
        print("Player2 won")
        
        


if __name__ == '__main__':
    main()
