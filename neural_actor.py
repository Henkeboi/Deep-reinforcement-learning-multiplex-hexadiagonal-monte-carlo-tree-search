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
        super(Network, self).__init__()
        self.output_size = output_size
        self.layers = torch.nn.ModuleList()
        layer = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2, 2))
        self.layers.append(layer)
        self.layers.append(torch.nn.Flatten())

        for i in range(0, len(dense_layers) - 1):
            input_size = dense_layers[i]
            layer = torch.nn.Linear(input_size, dense_layers[i + 1])
            self.layers.append(layer)
        model = self
        print(model)

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

class NeuralActor:
    def __init__(self, conv_layers, dense_layers, num_max_moves, learning_rate, device):
        torch.manual_seed(42)
        self.num_max_moves = num_max_moves
        self.nn = Network(conv_layers, dense_layers, num_max_moves)
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
                state = [x / sum(state) for x in state]
                state = torch.from_numpy(np.asarray(state)).float()
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
        while not StateManager.is_legal(move_index, state_str):
            nn_output.data[0, move_index] = -1.0
            move_index = torch.argmax(nn_output.data)
        return move_index.item()

    def store_model(self, name):
        torch.save(self.nn.state_dict(), 'data/' + name + '.pth')

    def load_model(self, name):
        self.nn.load_state_dict(torch.load('data/' + name + '.pth'))

def main():
    board_size = 4
    max_num_moves = int(board_size ** 2)
    state_space_size = int(board_size ** 2 + 1)
    conv_layers = []
    state_space_size = 9
    hidden_layers = [state_space_size, 25, max_num_moves] 
    la = 0.01

    state_manager = StateManager(board_size)
    num_search_games = 1
    num_simulations = 200

    if torch.cuda.is_available():
        print("Using gpu")
        device = 'cuda:0'
    else:
        print("Using cpu")
        device = 'cpu'
    device = 'cpu'

    player1 = NeuralActor(conv_layers, hidden_layers, max_num_moves, la, device)
    player2 = NeuralActor(conv_layers, hidden_layers, max_num_moves, la, device)
    mct1 = MCT(player1, num_search_games, num_simulations)
    mct2 = MCT(player2, num_search_games, num_simulations)

    #losses = []
    #start_time = time.time()
    #for i in range(0, 100):
    #    mct2.play_game(copy.deepcopy(state_manager))
    #    training_data = mct2.get_training_data()
    #    loss = player2.update_Q(training_data)
    #    losses.append(loss)
    #    print(str(i) + " " +  str(loss))
    #print(time.time() - start_time)
    #player2.store_model('16.1')
    player2.load_model('16.1')

    win1 = 0
    win2 = 0
    for i in range(0, 1000):
        state_manager = StateManager(board_size)
        while not state_manager.player1_won() and not state_manager.player2_won():
            if not state_manager.player1_to_move:
                move_index = random.randrange(0, board_size ** 2)
                while not StateManager.is_legal(move_index, state_manager.string_representation()):
                   move_index = random.randrange(0, board_size ** 2)
                #move = state_manager.convert_to_move(move_index)
                move = state_manager.convert_to_move(player2.get_action(state_manager.string_representation()))
            else:
                move_index = random.randrange(0, board_size ** 2)
                while not StateManager.is_legal(move_index, state_manager.string_representation()):
                   move_index = random.randrange(0, board_size ** 2)
                move = state_manager.convert_to_move(move_index)
                #move = state_manager.convert_to_move(player2.get_action(state_manager.string_representation()))
            state_manager.make_move(move)
            #state_manager.show()
        if state_manager.player1_won():
            win1 += 1
        elif state_manager.player2_won():
            win2 += 1
        else:
            print("No winner")

    print("Times player 1 won: " + str(win1) + ". " + "Times player2 won: " + str(win2))

    #state_manager.show()
    #if state_manager.player1_won():
    #    print("Player1 won")
    #elif state_manager.player2_won():
    #    print("Player2 won")

if __name__ == '__main__':
    main()
