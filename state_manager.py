class StateManager:
    def __init__(self, pieces_on_the_board=None, max_removable=None, string_representation=None):
        if string_representation == None:
            self.N = pieces_on_the_board
            self.K = max_removable
            self._player1_to_move = True
        else:
            list_rep = string_representation.split('.')
            self.N = int(list_rep[0])
            self.K = int(list_rep[1])
            if int(list_rep[2]):
                self._player1_to_move = True
            else:
                self._player1_to_move = False

    def get_moves(self):
        moves = []
        for i in range(1, self.K + 1):
            if i > self.N:
                return moves
            else:
                moves.append(i)
        return moves

    def make_move(self, move):
        assert(self.N >= move)
        assert(self.K >= move)
        if self._player1_to_move == False:
           self._player1_to_move = True 
        else:
           self._player1_to_move = False
        self.N = self.N - move

    def undo_move(self, move):
        assert(self.K >= move)
        if self._player1_to_move == False:
           self._player1_to_move = True 
        else:
           self._player1_to_move = False
        self.N = self.N + move

    def is_finished(self):
        if self.N == 0:
            return True
        return False
    
    def player1_to_move(self):
       return self._player1_to_move 

    def player2_to_move(self):
        return not self._player1_to_move 
    
    def player1_won(self):
        return not self._player1_to_move

    def player2_won(self):
        return self._player1_to_move

    def string_representation(self):
        string_rep = str(self.N) + '.' + str(self.K) + '.'
        if self._player1_to_move:
            string_rep = string_rep + str(1)
        else:
            string_rep = string_rep + str(0)
        return string_rep

def main():
    state = StateManager(4, 3)
    move = 1

    state.make_move(1)
    print(state.is_finished())
    state.make_move(2)
    print(state.is_finished())
    state.make_move(1)
    print(state.player2_won())
   
if __name__ == '__main__':
    main()
