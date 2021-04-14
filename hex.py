import networkx as nx
from matplotlib import pyplot as plt
import copy

class Cell:
    def __init__(self, value, x, y):
        self.value = value
        self.x = x
        self.y = y
        self.adjacent = []

    def get_neighbours(self):
        return self.adjacent

    def has_neighbour(self, cell):
        if cell in self.adjacent:
            return True
        else:
            return False

    def add_neighbour(self, cell):
        assert(not cell == None)
        assert(not self == cell)
        self.adjacent.append(cell)
        if not cell.has_neighbour(self):
            cell.add_neighbour(self)

class Hex:
    def __init__(self, size, string_rep=None):
        self.size = size 
        self.white_won = False
        self.black_won = False
        if string_rep == None:
            self.init_board()
        else:
            self.init_board_from_string(string_rep)
        self.debug_flag = False
    
    def get_upper_left(self, x, y):
        if y + 1 >= self.size:
            return None
        else:
            return self.board[y + 1][x]

    def get_upper_right(self, x, y): 
        if y + 1 >= self.size or x + 1 >= self.size:
            return None
        else:
            return self.board[y + 1][x + 1]

    def get_right(self, x, y): 
        if x + 1 >= self.size:
            return None
        else:
            return self.board[y][x + 1]

    def init_board(self):
        self.player1_to_move = True
        self.board = []
        for y in range(self.size):
            self.board.append([])
            for x in range(self.size):
                cell = Cell(0, x, y)
                self.board[y].append(cell)

        for y in range(self.size):
            for x in range(self.size):
                if (cell := self.get_right(x, y)) != None:
                    self.board[y][x].add_neighbour(cell)
                if (cell := self.get_upper_right(x, y)) != None:
                    self.board[y][x].add_neighbour(cell)
                if (cell := self.get_upper_left(x, y)) != None:
                    self.board[y][x].add_neighbour(cell)

    def init_board_from_string(self, string_rep):
        self.board = []
        if string_rep[-1] == str(1):
            self.player1_to_move = True
        else:
            self.player1_to_move = False

        string_index = 0
        for y in range(self.size):
            self.board.append([])
            for x in range(self.size):
                value = int(string_rep[string_index])
                cell = Cell(value, x, y)
                self.board[y].append(cell)
                string_index = string_index + 1

        for y in range(self.size):
            for x in range(self.size):
                if (cell := self.get_right(x, y)) != None:
                    self.board[y][x].add_neighbour(cell)
                if (cell := self.get_upper_right(x, y)) != None:
                    self.board[y][x].add_neighbour(cell)
                if (cell := self.get_upper_left(x, y)) != None:
                    self.board[y][x].add_neighbour(cell)

    def show(self):
        node_counter = 0
        pos = {}
        edges = []
        colors = []

        for x in range(0, self.size):
            for y in range(0, self.size):
                #pos_x = x - 0.5 * y
                #pos_y = y
                pos_x = (x + y) / (2 ** 0.5)
                pos_y = (y - x) / (2 ** 0.5)
                pos[node_counter] = (pos_x, pos_y)
                for neighbour in self.board[y][x].get_neighbours():
                    edges.append((node_counter, self.get_graph_index(neighbour.x, neighbour.y)))
                if self.board[y][x].value == 1:
                    colors.append('red')
                elif self.board[y][x].value == 0:
                    colors.append('green')
                elif self.board[y][x].value == 2:
                    colors.append('black')
                node_counter = node_counter + 1

        G = nx.Graph()
        G.add_nodes_from(pos.keys())
        plt.clf()
        nx.draw(G, pos, edgelist=edges, node_color=colors)
        plt.show(block=True)
        #plt.show(block=False)
        #plt.pause(3)

    def make_move(self, move):
        x = move[0]
        y = move[1]
        assert(self.board[y][x].value == 0)

        if self.player1_to_move:
            self.board[y][x].value = 1
            self.player1_to_move = False 
        else:
            self.board[y][x].value = 2
            self.player1_to_move = True

    def undo_move(self, move):
        x = move[0]
        y = move[1]
        assert(not self.board[y][x].value == 0)
        if self.player1_to_move == False:
            self.player1_to_move = True
        else:
            self.player1_to_move = False
        self.board[y][x].value = 0

    def get_moves(self):
        moves = []
        for y in range(0, self.size):
            for x in range(0, self.size):
                if self.board[y][x].value == 0:
                    moves.append((x, y))
        return moves

    def get_graph_index(self, x, y):
        return self.size * x + y

    def string_representation(self):
        string_rep = ''
        for y in range(self.size):
            for x in range(self.size):
                string_rep = string_rep + str(self.board[y][x].value)

        if self.player1_to_move == True:
            string_rep = string_rep + str(1)
        else:
            string_rep = string_rep + str(2)
        return string_rep

    def convert_state(self, state_tuple): # Needs only to move the turn indicator to the back
        state_str = ''
        for i in range(len(state_tuple) - 1):
            state_str += str(state_tuple[i + 1])
        state_str += str(state_tuple[0])
        return state_str

    def dfs_white(self, node, visited_nodes):
        visited_nodes.append(self.get_graph_index(node.x, node.y))
        for adjacent in node.get_neighbours():
            if self.get_graph_index(adjacent.x, adjacent.y) not in visited_nodes and adjacent.value == 1:
                if adjacent.y == self.size - 1:
                    self.white_won = True
                    return True
                self.dfs_white(adjacent, visited_nodes)
        return False

    def dfs_black(self, node, visited_nodes):
        visited_nodes.append(self.get_graph_index(node.x, node.y))
        for adj in node.get_neighbours():
            if self.get_graph_index(adj.x, adj.y) not in visited_nodes and adj.value == 2:
                if adj.x == self.size - 1:
                    self.black_won = True
                    return True
                self.dfs_black(adj, visited_nodes)
        return False


    def player1_won(self):
        visited_nodes = []
        start_nodes = []
        for x in range(self.size):
            if self.board[0][x].value == 1:
                start_nodes.append(self.board[0][x])

        for node in start_nodes:
            if self.get_graph_index(node.x, node.y) not in visited_nodes:
                self.dfs_white(node, visited_nodes)
                if self.white_won == True:
                    return True
        return False

    def player2_won(self):
        nodes_visited = []
        start_nodes = []
        for y in range(self.size):
            if self.board[y][0].value == 2:
                start_nodes.append(self.board[y][0])

        for node in start_nodes:
            self.dfs_black(node, nodes_visited)
            if self.black_won == True:
                return True
        return False
               
    def is_finished(self):
        if len(self.get_moves()) == 0:
            return True
        else:
            return False

    @staticmethod
    def is_legal(move, state):
        if state[move] == str(0):
            return True
        return False
       
    def convert_to_move(self, index):
        x = 0
        y = 0
        for i in range(index):
            x = x + 1
            if (x % self.size) == 0:
                y = y + 1
                x = 0
        return (x, y)
