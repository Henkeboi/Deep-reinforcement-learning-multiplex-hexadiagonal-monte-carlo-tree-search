from config import Config
from hex import Hex
from neural_actor import NeuralActor
from mct import MCT
import random
import math
import copy

class TOPP:
    def __init__(self):
        config = Config()
        config_dict = config.get_config()
        self.board_size = config_dict['board_size'] 
        self.num_episodes = config_dict['num_episodes']
        self.num_simulations = config_dict['simulations']
        self.la = config_dict['la']
        self.optimizer = config_dict['optimizer']
        self.M = config_dict['M']
        assert(self.M > 1)
        self.G = config_dict['G']
        config_dense_layers = config_dict['dense_layers']
        self.activation_functions = config_dict['activation_functions']
        self.train = config_dict['train']
        self.num_display = config_dict['num_display']
        self.max_num_moves = int(self.board_size ** 2)
        self.dense_layers = []
        self.dense_layers.append(int(self.board_size ** 2 + 1))
        for neurons in config_dense_layers:
            self.dense_layers.append(neurons)
        self.dense_layers.append(self.max_num_moves)
        self.players = []

    def play(self, player1, player2, G, board_size, num_display):
        win1 = 0
        win2 = 0
        player1_starting = True
        for i in range(G):
            state_manager = Hex(board_size)
            while not state_manager.player1_won() and not state_manager.player2_won():
                if state_manager.player1_to_move:
                    if player1_starting == True:
                        move = state_manager.convert_to_move(player1.get_action(state_manager.string_representation(), True))
                    else:
                        move = state_manager.convert_to_move(player2.get_action(state_manager.string_representation(), True))
                else:
                    if player1_starting  == True:
                        move = state_manager.convert_to_move(player2.get_action(state_manager.string_representation(), True))
                    else:
                        move = state_manager.convert_to_move(player1.get_action(state_manager.string_representation(), True))
                state_manager.make_move(move)
                if num_display > 0:
                    state_manager.show()

            num_display -= 1
            if state_manager.player1_won():
                if player1_starting:
                    win1 += 1
                else:
                    win2 += 1
            elif state_manager.player2_won():
                if player1_starting:
                    win2 += 1
                else:
                    win1 += 1
            else:
                print("No winner")
            player1_starting = not player1_starting
        return win1, win2

    def round_robin(self):
        if self.train == 1:
            players_to_load = self.M
            for i in range(self.num_episodes + 1):
                if i % math.floor(self.num_episodes / (self.M - 1)) == 0 and players_to_load > 0:
                    player = NeuralActor(self.dense_layers, self.activation_functions, self.max_num_moves, self.la, self.optimizer)
                    print("Load model " + str(i))
                    player.load_model('iteration' + str(i))
                    self.players.append(player)
                    players_to_load -= 1
        
        player_scores = [0 for i in range(len(self.players))]
        for i in range(len(self.players)):
            for j in range(i, len(self.players)):
                if not i == j:
                    print(str(i) + " vs " + str(j)) 
                    score1, score2 = self.play(self.players[i], self.players[j], self.G, self.board_size, self.num_display)
                    player_scores[i] += score1
                    player_scores[j] += score2
                    if self.num_display > 0:
                        self.num_display -= self.G
        
        for i in range(len(player_scores)):
            print("Score player" + str(i) + ": " + str(player_scores[i]))

    def do_training(self):
        self.players = []
        state_manager = Hex(self.board_size)
        player = NeuralActor(self.dense_layers, self.activation_functions, self.max_num_moves, self.la, self.optimizer)
        mct = MCT(player, self.num_episodes, self.num_simulations)

        # Train progressive policies
        players_to_store = self.M
        for i in range(0, self.num_episodes + 1):
            if i % math.floor(self.num_episodes / (self.M - 1)) == 0 and players_to_store > 0:
                print("Storing models/iteration" + str(i))
                player.store_model('iteration' + str(i))
                print("Stored")
                players_to_store -= 1
            loss = mct.play_game(copy.deepcopy(state_manager))
            print(str(i) + ": Loss : " + str(loss))

    def load_trained_players(self):
        self.players = []
        for i in range(1, 5):
            player = NeuralActor(self.dense_layers, self.activation_functions, self.max_num_moves, self.la, self.optimizer)
            print("Loading player" + str(i))
            player.load_model('player' + str(i))
            self.players.append(player)


