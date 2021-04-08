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
        for i in range(0, len(hidden_layers) - 1):
            input_size = hidden_layers[i]
            layer = torch.nn.Linear(input_size, hidden_layers[i + 1])
            self.layers.append(layer)

    def forward(self, tensor):
        output = None
        for i, layer in enumerate(self.layers):
            tensor = torch.relu(layer(tensor))
        output = torch.nn.functional.softmax(tensor, dim=0)
        return output
        

class NeuralActor:
    def __init__(self, hidden_layers, num_max_moves, learning_rate, device):
        torch.manual_seed(42)
        self.num_max_moves = num_max_moves
        self.nn = Network(hidden_layers, num_max_moves)
        self.dev = device
        self.nn.to(torch.device(self.dev))
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss() 
        self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate)
   
    def update_Q(self, training_data):
        loss = 0
        for epoch in range(1):
            for i in range(0, len(training_data)):
                state, label = training_data[i]
                state = torch.from_numpy(state).float()
                state = state.to(self.dev)
                nn_output = self.nn(state) # Forward pass
                nn_output = nn_output.view(1, self.num_max_moves)
                index = [x / sum(label) for x in label]
                label = torch.from_numpy(np.asarray([index])).type(torch.FloatTensor)
                label = label.to(self.dev)
                nn_loss = self.loss_function(nn_output, label)
                nn_loss.backward()
                loss += nn_loss.item()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def get_action(self, state_str):
        state = np.fromstring(state_str, np.int8) - 48
        state = torch.from_numpy(state).float().to(self.dev)
        nn_output = self.nn(state) # Forward pass
        move_index = torch.argmax(nn_output.data)
        while not Hex.is_legal(move_index, state_str):
            nn_output.data[move_index] = 0.0
            move_index = torch.argmax(nn_output.data)
        return move_index.item()

    def store_model(self, name):
        torch.save(self.nn.state_dict(), 'data/' + name + '.pth')

    def load_model(self, name):
        self.nn.load_state_dict(torch.load('data/' + name + '.pth'))

def main():
    board_size = 3
    max_num_moves = int(board_size ** 2)
    state_space_size = int(board_size ** 2 + 1)
    hidden_layers = [state_space_size, 100, max_num_moves] 
    la = 0.1

    state_manager = Hex(board_size)
    num_search_games = 1
    num_simulations = 20
    max_depth = 20

    if torch.cuda.is_available():
        print("Using gpu")
        device = 'cuda:0'
    else:
        print("Using cpu")
        device = 'cpu'
    device = 'cpu'

    player1 = NeuralActor(hidden_layers, max_num_moves, la, device)
    player2 = NeuralActor(hidden_layers, max_num_moves, la, device)
    mct1 = MCT(player1, num_search_games, num_simulations, max_depth)
    mct2 = MCT(player2, num_search_games, num_simulations, max_depth)

    #for i in range(0, 1):
    #    mct1.play_game(copy.deepcopy(state_manager))
    #    training_data = mct1.get_training_data()
    #    player1.update_Q(training_data)

    #losses = []
    #for i in range(0, 10000):
    #    mct2.play_game(copy.deepcopy(state_manager))
    #    training_data = mct2.get_training_data()
    #    loss = player2.update_Q(training_data)
    #    losses.append(loss)
    #    print(str(i) + " " +  str(loss))
    player2.load_model('player2')
    #player2.store_model('player2')
    #plt.plot(losses)
    #plt.show()

    while not state_manager.player1_won() and not state_manager.player2_won():
        if state_manager.player1_to_move:
            print("Player 1 moved")
            move = state_manager.convert_to_move(player1.get_action(state_manager.string_representation()))
        else:
            print("Player 2 moved")
            move = state_manager.convert_to_move(player2.get_action(state_manager.string_representation()))
        state_manager.make_move(move)
        state_manager.show()

    if state_manager.player1_won():
        print("Player1 won")
    elif state_manager.player2_won():
        print("Player2 won")

if __name__ == '__main__':
    main()
