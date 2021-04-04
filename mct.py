from node import Node
import numpy as np
import random
from operator import itemgetter
import copy
from hex import Hex

class MCT:
    def __init__(self, nn, num_search_games, num_simulations, max_depth):
        self.nn = nn
        self.num_search_games = num_search_games
        self.num_simulations = num_simulations
        self.max_depth = max_depth

    def traverse_to_leaf(self):
        parent = self.root_node
        depth = 0
        reached_final_state = False
        while depth < self.max_depth and not reached_final_state:
            children = parent.get_children()
            if len(children) == 0:
                reached_final_state = True
            else:
                selected_child = self.tree_policy_select(parent, children)
                parent.update_edge(selected_child.state)
                parent = selected_child
                depth = depth + 1
        return parent

    def backpropagate(self, leaf, score):
        while not leaf == None:
            leaf.update_Q(score)
            leaf = leaf.parent

    def rollout(self, leaf):
        size = int((len(leaf.state) - 1) ** 0.5)
        leaf_state = copy.deepcopy(Hex(size, leaf.state))
        first_iteration = True
        while True:
            if leaf_state.player1_won():
                score = 1.0
                return score
            elif leaf_state.player2_won(): 
                score = -1.0
                return score
            if leaf_state.is_finished():
                print("No winner error")
                print(leaf_state.player1_won())
                print(leaf_state.player2_won())
                quit()

            if first_iteration:
                possible_moves = leaf_state.get_moves()
                move = possible_moves[random.randint(0, len(possible_moves) - 1)]
                first_iteration = False
            else:
                possible_moves = leaf_state.get_moves() ## DEBUG
                move = possible_moves[random.randint(0, len(possible_moves) - 1)] ## DEBUG
                #possible_moves = leaf_state.get_moves()
                #move = possible_moves[self.nn.get_action(leaf_state.string_representation())]
            leaf_state.make_move(move)
       
    def tree_policy_select(self, parent, children):
        size = int((len(parent.state) - 1) ** 0.5)
        parent_state = copy.deepcopy(Hex(size, parent.state))
        return children[random.randint(0, len(children) - 1)] ## DEBUG
        if parent_state.player1_to_move:
            max_Q = 0.0
            selected_child = None
            for child in children:
                Q = child.get_Q(parent_state.player1_to_move()) + np.sqrt(np.log(parent.num_traversed) / (1.0 + parent.num_traversed_edge(child.state)))
                if Q > max_Q or selected_child == None:
                    max_Q = Q
                    selected_child = child
        else:
            max_Q = 0.0
            selected_child = None
            for child in children:
                Q = child.get_Q(parent_state.player1_to_move()) + np.sqrt(np.log(parent.num_traversed) / (1.0 + parent.num_traversed_edge(child.state)))
                if Q < max_Q or selected_child == None:
                    max_Q = Q
                    selected_child = child
        return selected_child

    def run_simulations(self):
        for i in range(0, self.num_simulations):
            leaf = copy.deepcopy(self.traverse_to_leaf())
            score = self.rollout(leaf)
            self.backpropagate(leaf, score)

    def play_game(self, root_state):
        self.root_node = Node(None, copy.deepcopy(root_state.string_representation()))
        game_finished = False
        i = 0
        while not game_finished:
            self.run_simulations()
            if len(self.root_node.children) == 0:
                game_finished = True
            else:
                self.root_node = max(self.root_node.children, key=itemgetter(1))[0]

    def create_tensor(self, string):
        state = np.fromstring(string, dtype=int, sep=".")
        return state

    def get_training_data(self):
        training_data = []
        self.root_node = self.root_node.parent # The last node has no label.
        while not self.root_node == None:
            state = self.create_tensor(self.root_node.state)
            label = [item[1] for item in self.root_node.children]
            training_data.append((state, label))
            self.root_node = self.root_node.parent
        return training_data

def main():
    num_search_games = 1
    num_simulations = 1
    max_depth = 100
    mct1 = MCT(None, num_search_games, num_simulations, max_depth)

    state_manager = Hex(4)
    for i in range(0, 1):
        mct1.play_game(copy.deepcopy(state_manager))
    
    
if __name__ == '__main__':
    main()
