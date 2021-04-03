import networkx as nx
from matplotlib import pyplot as plt

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
        if string_rep == None:
            self.init_board()
        else:
            self.init_board_from_string(string_rep)
    
    def get_left(self, x, y):
        if x - 1 < 0:
            return None
        else:
            return self.board[y][x - 1]

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

    def get_lower_right(self, x, y):
        if x + 1 >= self.size or y - 1 < 0:
            return None
        else:
            return self.board[y - 1][x + 1]

    def get_lower_left(self, x, y):
        if x - 1 < 0 or y - 1 < 0:
            return None
        else:
            return self.board[y - 1][x - 1]

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
                pos_x = x - 0.5 * y
                pos_y = y
                pos[node_counter] = (pos_x, pos_y)
                for neighbour in self.board[y][x].get_neighbours():
                    edges.append((node_counter, self.get_graph_index(neighbour.x, neighbour.y)))
                if self.board[y][x].value == 1:
                    colors.append('red')
                elif self.board[y][x].value == 0:
                    colors.append('green')
                elif self.board[y][x].value == 2:
                    colors.append('blue')
                node_counter = node_counter + 1

        G = nx.Graph()
        G.add_nodes_from(pos.keys())
        plt.clf()
        nx.draw(G, pos, edgelist=edges, node_color=colors)
        plt.show(block=False)
        plt.pause(500)

    def make_move(self, move):
        x = move[0]
        y = move[1]
        assert(self.board[y][x].value == 0)
        self.player1_to_move = False ### Debug
        if self.player1_to_move:
            self.board[y][x].value = 1
            self.player1_to_move = False 
        else:
            self.board[y][x].value = 2
            self.player1_to_move = True

    def get_moves(self):
        moves = []
        for y in range(0, self.size):
            for x in range(0, self.size):
                if self.board[y][x] == 0:
                    moves.append(x, y)
        return moves

    def undo_move(self, move):
        assert(not self.board[y][x] == 0)
        if self._player1_to_move == False:
            self._player1_to_move = True
        else:
            self._player1_to_move = False
        self.board[y][x] = 0

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
            string_rep = string_rep + str(0)
        return string_rep


    def dfs_util(self, node, visited_nodes, color):
        visited_nodes.append(self.get_graph_index(node.x, node.y))
        if color == 'white':
            for adjacent in node.get_neighbours():
                if adjacent.value == 1:
                    if node.x == self.size - 1:
                        print("White won")
                    if self.get_graph_index(adjacent.x, adjacent.y) not in visited_nodes:
                        self.dfs_util(adjacent, visited_nodes, color)
            
        elif color == 'black':
            for adjacent in node.get_neighbours():
                if adjacent.value == 2:
                    #print("x: " + str(node.x))
                    #print("y: " + str(node.y))
                    #print()
                    if node.y == self.size - 1:
                        print("Black won")
                    if self.get_graph_index(adjacent.x, adjacent.y) not in visited_nodes:
                        self.dfs_util(adjacent, visited_nodes, color)

    def check_if_white_won(self):
        visited_nodes = []
        start_nodes = []
        goal_nodes = []
        for y in range(self.size):
            if self.board[y][0].value == 1:
                start_nodes.append(self.board[y][0])
            if self.board[y][self.size - 1].value == 1:
                goal_nodes.append(self.board[y][self.size - 1])

        for node in start_nodes:
            if self.get_graph_index(node.x, node.y) not in visited_nodes:
                self.dfs_util(node, visited_nodes, 'white')

    def check_if_black_won(self):
        visited_nodes = []
        start_nodes = []
        goal_nodes = []
        for x in range(self.size):
            if self.board[0][x].value == 2:
                start_nodes.append(self.board[0][x])
            if self.board[self.size - 1][x].value == 2:
                goal_nodes.append(self.board[self.size - 1][x])

        for node in start_nodes:
            if self.get_graph_index(node.x, node.y) not in visited_nodes:
                self.dfs_util(node, visited_nodes, 'black')

    def is_finished(self):
        self.check_if_white_won()
        self.check_if_black_won()
               
def main():
    hex_board = Hex(4)
    rep = hex_board.string_representation()
    hex_board = Hex(hex_board.size, string_rep=rep)
    move = (0, 0)
    hex_board.make_move(move)
    move = (1, 1)
    hex_board.make_move(move)
    move = (1, 2)
    hex_board.make_move(move)
    move = (1, 3)
    hex_board.make_move(move)
    hex_board.is_finished()
    hex_board.show()    

if __name__ == '__main__':
    main()
