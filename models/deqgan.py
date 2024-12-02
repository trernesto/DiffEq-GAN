import torch.nn as nn
import torch 

class Generator(nn.Module):
    def __init__(self, input_size = 1, hidden_layer_size = 12, number_of_hidden_layers = 2, output_size = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.output_size = output_size

        self.layers = nn.Sequential()

        in_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_size),
            nn.Tanh()
        )

        self.layers.add_module('input layer', in_layer)

        hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.Tanh()
        )

        for layer_number in range(self.number_of_hidden_layers - 1):
            self.layers.add_module(f'hidden layer № {layer_number + 1}', hidden_layer)

        out_layer = hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.output_size)
        )

        self.layers.add_module('output layer', out_layer)

    def forward(self, x: torch.tensor):
        out = self.layers(x)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_size = 1, hidden_layer_size = 12, number_of_hidden_layers = 2, output_size = 1):
    
        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.output_size = output_size

        self.layers = nn.Sequential()

        in_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_size),
            nn.Tanh()
        )

        self.layers.add_module('input layer', in_layer)

        hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.Tanh()
        )

        for layer_number in range(self.number_of_hidden_layers - 1):
            self.layers.add_module(f'hidden layer № {layer_number + 1}', hidden_layer)

        out_layer = hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.output_size),
            nn.Sigmoid()
        )

        self.layers.add_module('output layer', out_layer)

    def forward(self, x: torch.tensor):
        out = self.layers(x)
        return out