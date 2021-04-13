from config import Config
from state_manager import StateManager
from neural_actor import NeuralActor
from mct import MCT
import copy

def main():
    config = Config()
    config_dict = config.get_config()
    board_size = config_dict['board_size'] 
    episodes = config_dict['episodes']
    num_simulations = config_dict['simulations']
    la = config_dict['la']
    optimizer = config_dict['optimizer']
    M = config_dict['M']
    G = config_dict['G']
    conv_layer_filters = config_dict['conv_layer_filters']
    conv_layer_kernels = config_dict['conv_layer_kernels']

    max_num_moves = int(board_size ** 2)
    state_space_size = int(board_size ** 2 + 1)
    conv_layers = (conv_layer_filters, conv_layer_kernels)
    state_space_size = 128
    hidden_layers = [state_space_size, max_num_moves]

    state_manager = StateManager(board_size)
    player = NeuralActor(conv_layers, hidden_layers, max_num_moves, la, optimizer)
    mct = MCT(player, num_simulations)

    # Train progressive policies
    for i in range(episodes):
        mct.play_game(copy.deepcopy(state_manager))
        training_data = mct.get_training_data()
        loss = player.update_Q(training_data)
        print(str(i) + " " +  str(loss))
        if i % (episodes // (M - 1)) == 0:
            player.store_model('models/iteration' + str(i))
    player.store_model('models/iteration' + str(episodes))




if __name__ == '__main__':
    main()
