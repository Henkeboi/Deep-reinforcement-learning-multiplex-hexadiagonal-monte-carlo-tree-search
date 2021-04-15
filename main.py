from config import Config
from hex import Hex
from neural_actor import NeuralActor
from mct import MCT
import random
import math
import copy
from topp import TOPP

def main():
    config = Config()
    config_dict = config.get_config()
    board_size = config_dict['board_size'] 
    num_episodes = config_dict['num_episodes']
    num_simulations = config_dict['simulations']
    la = config_dict['la']
    optimizer = config_dict['optimizer']
    M = config_dict['M']
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

    topp = TOPP() 
    topp.play_random()
    quit()

    if not train == 1:
        print("Using stored models")
        topp.load_trained_players()
        topp.round_robin()
    if train == 1:
        topp.do_training()
        topp.round_robin()

if __name__ == '__main__':
    main()
