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
from network import Network

class NeuralActor:
    def __init__(self, conv_layers, dense_layers, num_max_moves, learning_rate, optimizer):
        torch.manual_seed(42)
        self.num_max_moves = num_max_moves
        self.nn = Network(conv_layers, dense_layers, num_max_moves)
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss() 
        if optimizer.lower() == 'adagrad':
            self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.nn.parameters(), lr=self.learning_rate)
        elif optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)

   
    def update_Q(self, training_data):
        loss = 0
        for i in range(0, len(training_data)):
            state, label = training_data[i]
            state = [x / sum(state) for x in state]
            state = torch.from_numpy(np.asarray(state)).float()
            nn_output = self.nn(state) # Forward pass
            nn_output = nn_output.view(1, self.num_max_moves)
            index = [x / sum(label) for x in label]
            label = torch.from_numpy(np.asarray([index])).type(torch.FloatTensor)
            nn_loss = self.loss_function(nn_output, label)
            nn_loss.backward()
            loss += nn_loss.item()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def get_action(self, state_str):
        state = np.fromstring(state_str, np.int8) - 48
        state = torch.from_numpy(state).float()
        nn_output = self.nn(state) # Forward pass
        move_index = torch.argmax(nn_output.data)
        while not StateManager.is_legal(move_index, state_str):
            nn_output.data[0, move_index] = -1.0
            move_index = torch.argmax(nn_output.data)
        return move_index.item()

    def store_model(self, name):
        torch.save(self.nn.state_dict(), name + '.pth')

    def load_model(self, name):
        self.nn.load_state_dict(torch.load(name + '.pth'))

def main():
    board_size = 4
    max_num_moves = int(board_size ** 2)
    state_space_size = int(board_size ** 2 + 1)
    conv_layers = []
    state_space_size = 128
    hidden_layers = [state_space_size, max_num_moves] 
    la = 0.01

    state_manager = StateManager(board_size)
    num_simulations = 200

    player1 = NeuralActor(conv_layers, hidden_layers, max_num_moves, la, 'sgd')
    player2 = NeuralActor(conv_layers, hidden_layers, max_num_moves, la, 'sgd')
    mct1 = MCT(player1, num_simulations)
    mct2 = MCT(player2, num_simulations)

    train = False
    if train == True:
        start_time = time.time()
        for i in range(0, 100):
            mct1.play_game(copy.deepcopy(state_manager))
            training_data = mct1.get_training_data()
            loss = player1.update_Q(training_data)
            print(str(i) + " " +  str(loss))
        player1.store_model('data/16.3')
    else:
        player1.load_model('data/16.3')
        player2.load_model('data/16.3')

    win1 = 0
    win2 = 0
    for i in range(0, 1000):
        state_manager = StateManager(board_size)
        while not state_manager.player1_won() and not state_manager.player2_won():
            if not state_manager.player1_to_move:
                move_index = random.randrange(0, board_size ** 2)
                while not StateManager.is_legal(move_index, state_manager.string_representation()):
                   move_index = random.randrange(0, board_size ** 2)
                move = state_manager.convert_to_move(move_index)
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


if __name__ == '__main__':
    main()
