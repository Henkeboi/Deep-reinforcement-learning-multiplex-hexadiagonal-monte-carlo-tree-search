from state_manager import StateManager
import random
import numpy as np
from node import Node

class MCT:
    def __init__(self, root_state, num_search_games, num_simulations, max_depth):
        self.root_node = Node(None, root_state.string_representation())
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

    def run_simulations(self):
        for i in range(0, self.num_simulations):
            leaf = self.traverse_to_leaf()
            print(self.root_node.Q)
            score = self.rollout(leaf)
            self.backpropagate(leaf, score)

    def rollout(self, leaf):
        leaf_state = StateManager(None, None, leaf.state)
        while not leaf_state.is_finished():
            possible_moves = leaf_state.get_moves()
            move = possible_moves[random.randint(0, len(possible_moves) - 1)]
            leaf_state.make_move(move)
        if leaf_state.player1_won():
            score = 1.0
        elif leaf_state.player2_won():
            score = -1.0
        else:
            print("Rollout error")
            quit()
        return score
        
    def tree_policy_select(self, parent, children):
        parent_state = StateManager(None, None, parent.state)
        if parent_state.player1_to_move():
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

def main():
    number_of_pieces = 15
    max_removable_pieces = 3
    state_manager = StateManager(number_of_pieces, max_removable_pieces)

    num_search_games = 1
    num_simulations = 1000
    max_depth = 15
    mct = MCT(state_manager, num_search_games, num_simulations, max_depth)
    mct.run_simulations()



if __name__ == '__main__':
    main()

