import re

class Config:
    def __init__(self):
        self.data = {
            'board_size' : None,
            'num_episodes' : None,
            'simulations' : None,
            'la' : None,
            'optimizer' : None,
            'M': None,
            'G': None,
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
        if variable == 'num_episodes':
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
        if variable == 'dense_layers':
            self.data['dense_layers'] = []
            for var in data.split(','):
                self.data['dense_layers'].append(int(var))

