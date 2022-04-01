import torch
import torch.nn as nn
from neural_rendering.utils import *


def fc_layer(in_features, out_features):
    net = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True)
    )

    return net


class PositionalPixelGenerator(nn.Module):

    def __init__(self, buffers_features, variables_features, out_features=3, hidden_features=700, hidden_layers=8, device='cuda', sine=False):
        super(PositionalPixelGenerator, self).__init__()

        self.buffers_features = buffers_features
        self.variables_features = variables_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        self.inner_pos = fc_layer(in_features=3, out_features=hidden_features)

        self.inner = fc_layer(in_features=hidden_features + buffers_features + variables_features, out_features=hidden_features)

        self.hidden = nn.ModuleList()
        for i in range(hidden_layers):
            self.hidden.append(fc_layer(in_features=(hidden_features + buffers_features + variables_features), out_features=hidden_features))

        self.outer = nn.Linear(in_features=hidden_features + buffers_features + variables_features, out_features=out_features)

        print("Number of model parameters:")
        print_network(self)

    def forward(self, input):
        # Get emission and position from input
        emission = input[:, :, :, 0:3]
        position = input[:, :, :, 6:9]

        # Ignore emission, it's passed through to the output
        input = input[:, :, :, 3:]

        # Positional encoding
        x0 = self.inner_pos(position)
        x0 = torch.cat([x0, input], 3)

        x1 = self.inner(x0)
        prev = x1

        for i in range(len(self.hidden)):
            x2 = torch.cat([prev, input], 3)
            x2 = self.hidden[i](x2)
            prev = x2

        x2 = torch.cat([prev, input], 3)
        output = self.outer(x2)

        # Merge emission and predicted output
        output = torch.where(emission > 1.0, emission, output+emission)

        return output