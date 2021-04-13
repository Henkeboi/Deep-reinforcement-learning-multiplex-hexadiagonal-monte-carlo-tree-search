import re

class Config:
    def __init__(self):
        self.data = {
            'board_size' : None,
            'episodes' : None,
            'simulations' : None,
            'la' : None,
            'optimizer' : None,
            'M': None,
            'G': None,
            'conv_layer_filters' : None,
            'conv_layer_kernels' : None,
            'dense_layers' : None
       }
    
    def get_config(self):
        self.read_config()
        return self.data

    def read_config(self):
        with open("./config_file.txt", "r") as f:
            config_data = f.readlines()

        for line in config_data:
            line = line.strip("\n")
            variable, value = line.split(":")
            self.parse_data(variable, value)
    
    def parse_data(self, variable, data):
        if variable == 'board_size':
            self.data[variable] = int(data)
        if variable == 'episodes':
            self.data[variable] = int(data)
        if variable == 'simulations':
            self.data[variable] = int(data)
        if variable == 'la':
            self.data[variable] = float(data)
        if variable == 'optimizer':
            self.data[variable] = data
        if variable == 'M':
            self.data[variable] = int(data)
        if variable == 'G':
            self.data[variable] = int(data)
        if variable == 'conv_layers':
            self.data['conv_layer_filters'] = []
            self.data['conv_layer_kernels'] = []
            for var in data.split(','):
                layer = var.replace('(', '').replace(')', '').split(' ')
                self.data['conv_layer_filters'].append(int(layer[0]))
                self.data['conv_layer_kernels'].append((int(layer[1]), int(layer[2])))

