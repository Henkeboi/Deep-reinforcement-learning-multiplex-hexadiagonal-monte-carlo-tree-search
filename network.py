import random
import copy
import operator
import torch

class Network(torch.nn.Module):
    def __init__(self, board_size, hidden_layers):
        super(Network, self).__init__()
        self._layers = torch.nn.ModuleList()
        input_size = board_size
        # Hidden layers
        if not hidden_layers[0] == 0:
            for i in range(len(hidden_layers)):
                output_size = hidden_layers[i]
                self._layers.append(torch.nn.Linear(input_size, output_size))
                input_size = output_size

        # Output layer
        output_size = 1
        self._hidden_layer = torch.nn.Linear(input_size, output_size)

    def forward(self, tensor):
        output = None
        for layer in reversed(self._layers):
            output = torch.(layer(tensor))
            tensor = output
        output = self._hidden_layer(tensor)
        return output

class NeuralCritic:
    def __init__(self, board_size, learning_rate, discount, trace_decay, hidden_layers):
        torch.manual_seed(42)
        self._learning_rate = learning_rate
        self._discount = discount
        self._loss_function = torch.nn.MSELoss()
        self._nn = Network(board_size, hidden_layers)
        self._optimizer = torch.optim.SGD(self._nn.parameters(), lr=self._learning_rate)
   
    def update_Q(self, old_board, new_board):
        TD_error = self.get_TD_error(old_board, new_board)
        old_board_tensor = torch.from_numpy(old_board).type(torch.FloatTensor)
        nn_output = self._nn(old_board_tensor) # Forward pass

        # Update the gradients by the size of td error
        nn_loss = self._loss_function(TD_error + nn_output, nn_output)
        nn_loss.backward() 
        
        for i, forward in enumerate(self._nn.parameters()):
            self._trace[i] = self.get_trace_decay(i) + (2 * forward.grad) # Update eligibility
            forward.grad = float(TD_error) * self._trace[i] # Contribution to recent predecessor states
            self._trace[i] = self._discount * self._trace_decay * self._trace[i] # Eligiblties decay
        self._optimizer.step() # Update both
        self._optimizer.zero_grad()
    
    def get_TD_error(self, old_board, new_board):
        return self._discount * self.get_V(new_board) - self.get_V(old_board)

    def get_V(self, board):
        input_tensor = torch.from_numpy(board).type(torch.FloatTensor)
        output = self._nn(input_tensor)
        return output
