from state_manager import StateManager
import random
import numpy as np


class Node:
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.children = None # Stores tuples of children and times a child is visited
        self.num_traversed = 0
        self.Q = 0.0

    def is_expanded(self):
        if self.children == None:
            return False
        return True

    def update_edge(self, child_state):
        for i, child in enumerate(self.children):
            if child[0].state == child_state:
                self.children[i] = (self.children[i][0], self.children[i][1] + 1)
                return
        print("Error")

    def num_traversed_edge(self, child_state):
        if self.children == None:
            return 0.0
        for i, child in enumerate(self.children):
            if child[0].state == child_state:
                return self.children[i][1]
        assert(1 == 2)

    def expand(self):
        size = int((len(self.state) - 1) ** 0.5)
        state_manager = StateManager(size, self.state)
        possible_moves = state_manager.get_moves()
        self.children = []
        for move in possible_moves:
            state_manager.make_move(move)
            self.children.append((Node(self, state_manager.string_representation()), 0)) # Visited 0 times
            state_manager.undo_move(move)

    def get_children(self):
        self.num_traversed = self.num_traversed + 1
        if not self.is_expanded():
            self.expand()
        return [child[0] for child in self.children]

    def get_Q(self, player1_to_move):
        if self.num_traversed == 0:
            if player1_to_move:
                return np.inf
            else:
                return -np.inf
        else:
            return self.Q

    def update_Q(self, value):
        self.Q = self.Q + value
