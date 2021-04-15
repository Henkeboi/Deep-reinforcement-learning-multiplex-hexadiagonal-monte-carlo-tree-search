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
        board_size = config_dict['board_size'] 
        num_episodes = config_dict['num_episodes']
        num_simulations = config_dict['simulations']
        la = config_dict['la']
        optimizer = config_dict['optimizer']
        M = config_dict['M']
        assert(M > 1)
        G = config_dict['G']
        config_dense_layers = config_dict['dense_layers']
        activation_functions = config_dict['activation_functions']
        train = config_dict['train']
        num_display = config_dict['num_display']

        max_num_moves = int(board_size ** 2)


def play(player1, player2, G, board_size, num_display):
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


def train():
    config = Config()
    config_dict = config.get_config()
    board_size = config_dict['board_size'] 
    num_episodes = config_dict['num_episodes']
    num_simulations = config_dict['simulations']
    la = config_dict['la']
    optimizer = config_dict['optimizer']
    M = config_dict['M']
    assert(M > 1)
    G = config_dict['G']
    config_dense_layers = config_dict['dense_layers']
    activation_functions = config_dict['activation_functions']
    train = config_dict['train']
    num_display = config_dict['num_display']

    max_num_moves = int(board_size ** 2)
    dense_layers = []
    dense_layers.append(int(board_size ** 2 + 1))
    for neurons in config_dense_layers:
        dense_layers.append(neurons)
    dense_layers.append(max_num_moves)

    state_manager = Hex(board_size)

    player = NeuralActor(dense_layers, activation_functions, max_num_moves, la, optimizer)
    mct = MCT(player, num_episodes, num_simulations)

    # Train progressive policies
    for i in range(0, num_episodes + 1):
        if i % math.floor(num_episodes / M) == 0 and not i == 0:
            print("Storing models/iteration" + str(i))
            player.store_model('iteration' + str(i))
            print("Stored")
        mct.play_game(copy.deepcopy(state_manager))
        training_data = mct.get_training_data()
        loss = player.update_Q(training_data)
        print(str(i) + " " +  str(loss))


def main():
    config = Config()
    config_dict = config.get_config()
    board_size = config_dict['board_size'] 
    num_episodes = config_dict['num_episodes']
    num_simulations = config_dict['simulations']
    la = config_dict['la']
    optimizer = config_dict['optimizer']
    M = config_dict['M']
    assert(M > 1)
    G = config_dict['G']
    config_dense_layers = config_dict['dense_layers']
    activation_functions = config_dict['activation_functions']
    train = config_dict['train']
    num_display = config_dict['num_display']

    max_num_moves = int(board_size ** 2)
    dense_layers = []
    dense_layers.append(int(board_size ** 2 + 1))
    for neurons in config_dense_layers:
        dense_layers.append(neurons)
    dense_layers.append(max_num_moves)

    state_manager = Hex(board_size)

    player = NeuralActor(dense_layers, activation_functions, max_num_moves, la, optimizer)
    mct = MCT(player, num_episodes, num_simulations)

    # Train progressive policies
    if train == 1:
        for i in range(0, num_episodes + 1):
            if i % math.floor(num_episodes / M) == 0 and not i == 0:
                print("Storing models/iteration" + str(i))
                player.store_model('iteration' + str(i))
                print("Stored")
            mct.play_game(copy.deepcopy(state_manager))
            training_data = mct.get_training_data()
            loss = player.update_Q(training_data)
            print(str(i) + " " +  str(loss))

    players = []
    for i in range(num_episodes + 1):
        if i % math.floor(num_episodes / M) == 0 and not i == 0:
            player = NeuralActor(dense_layers, activation_functions, max_num_moves, la, optimizer)
            player.load_model('iteration' + str(i))
            players.append(player)
    
    player_scores = [0 for i in range(len(players))]
    for i in range(len(players)):
        for j in range(i, len(players)):
            if not i == j:
                print(str(i) + " vs " + str(j)) 
                score1, score2 = play(players[i], players[j], G, board_size, num_display)
                player_scores[i] += score1
                player_scores[j] += score2
                if num_display > 0:
                    num_display -= G
    
    for i in range(len(player_scores)):
        print("Score player" + str(i) + ": " + str(player_scores[i]))


if __name__ == '__main__':
    main()
