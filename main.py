from config import Config
from hex import Hex
from neural_actor import NeuralActor
from mct import MCT
import random
import math
import copy


def play(player1, player2):
    win1 = 0
    win2 = 0
    player1_starting = True
    for i in range(0, 2000):
        state_manager = Hex(4)
        while not state_manager.player1_won() and not state_manager.player2_won():
            if state_manager.player1_to_move:
                #move_index = random.randrange(0, board_size ** 2)
                #while not Hex.is_legal(move_index, state_manager.string_representation()):
                #    move_index = random.randrange(0, board_size ** 2)
                #move = state_manager.convert_to_move(move_index)
                if player1_starting == True:
                    move = state_manager.convert_to_move(player1.get_action(state_manager.string_representation(), True))
                else:
                    move = state_manager.convert_to_move(player2.get_action(state_manager.string_representation(), True))
            else:
                #move_index = random.randrange(0, board_size ** 2)
                #while not Hex.is_legal(move_index, state_manager.string_representation()):
                #    move_index = random.randrange(0, board_size ** 2)
                #move = state_manager.convert_to_move(move_index)
                if player1_starting  == True:
                    move = state_manager.convert_to_move(player2.get_action(state_manager.string_representation(), True))
                else:
                    move = state_manager.convert_to_move(player1.get_action(state_manager.string_representation(), True))
            state_manager.make_move(move)
            #state_manager.show()
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

    max_depth = 1000000000

    max_num_moves = int(board_size ** 2)
    dense_layers = []
    dense_layers.append(int(board_size ** 2 + 1))
    for neurons in config_dense_layers:
        dense_layers.append(neurons)
    dense_layers.append(max_num_moves)
    state_manager = Hex(board_size)
    #player = NeuralActor(dense_layers, max_num_moves, la, optimizer)
    #mct = MCT(player1, num_simulations)

    device = 'cpu'
    player = NeuralActor(dense_layers, max_num_moves, la, device)
    mct = MCT(player, num_episodes, num_simulations, max_depth)

    # Train progressive policies
    train = False
    if train == True:
        for i in range(num_episodes):
            if math.floor(num_episodes / (M - 1)) == 0:
                print("Modulo zero with this Episode and M config.")
                quit()
            else:
                if i % math.floor(num_episodes / M) == 0 and not i == 0:
                    print("Storing models/iteration" + str(i))
                    player.store_model('iteration' + str(i))
                    print("Stored")
            mct.play_game(copy.deepcopy(state_manager))
            training_data = mct.get_training_data()
            loss = player.update_Q(training_data)
            print(str(i) + " " +  str(loss))
        print("Storing models/iteration" + str(num_episodes))
        player.store_model('iteration' + str(num_episodes))
        print("Stored")

    players = []
    for i in range(num_episodes + 1):
        if i % math.floor(num_episodes / M) == 0 and not i == 0:
            player = NeuralActor(dense_layers, max_num_moves, la, device)
            player.load_model('iteration' + str(i))
            players.append(player)
    
    player_scores = [0 for i in range(len(players))]
    for i in range(len(players)):
        for j in range(i, len(players)):
            if not i == j:
                print(str(i) + " vs " + str(j)) 
                score1, score2 = play(players[i], players[j])
                player_scores[i] += score1
                player_scores[j] += score2
    
    for i in range(len(player_scores)):
        print("Score player" + str(i) + ": " + str(player_scores[i]))


if __name__ == '__main__':
    main()
