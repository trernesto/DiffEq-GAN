#Implementation of pinn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN():
    #input_size ~ size(len) of input vector
    def __init__(self, input_size = 1, hidden_layer_size = 12, number_of_hidden_layers = 2, output_size = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.output_size = output_size

        self.layers = nn.Sequential()

        in_layer = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_size),
            nn.ReLU()
        )

        self.layers.add_module('input layer', in_layer)

        hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.ReLU()
        )

        for layer_number in range(self.number_of_hidden_layers - 1):
            self.layers.add_module(f'hidden layer № {layer_number + 1}', hidden_layer)

        out_layer = hidden_layer = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.output_size)
        )

        self.layers.add_module('output layer', out_layer)

    def forward(self, x: nn.torch):
        out = self.layers(x)
        return out