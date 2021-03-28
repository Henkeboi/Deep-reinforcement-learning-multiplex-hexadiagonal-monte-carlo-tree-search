class StateManager:
    def __init__(self, pieces_on_the_board=None, max_removable=None, string_representation=None):
        if string_representation == None:
            self._N = pieces_on_the_board
            self._K = max_removable
            self._player1_to_move = True
        else:
            list_rep = string_representation.split('.')
            self._N = int(list_rep[0])
            self._K = int(list_rep[1])
            if int(list_rep[2]):
                self._player1_to_move = True
            else:
                self._player1_to_move = False

    def get_moves(self):
        moves = []
        for i in range(1, self._K + 1):
            if i > self._N:
                return moves
            else:
                moves.append(i)
        return moves

    def make_move(self, move):
        assert(self._N >= move)
        assert(self._K >= move)
        if self._player1_to_move == False:
           self._player1_to_move = True 
        else:
           self._player1_to_move = False
        self._N = self._N - move

    def undo_move(self, move):
        assert(self._K >= move)
        if self._player1_to_move == False:
           self._player1_to_move = True 
        else:
           self._player1_to_move = False
        self._N = self._N + move

    def is_finished(self):
        if self._N == 0:
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
        string_rep = str(self._N) + '.' + str(self._K) + '.'
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
