import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Foo:
    def __init__(self):
        self.na = "batman"

    def foo(self):
        print(self.na)


class GRUBasic(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(GRUBasic, self).__init__()
        self.gru = nn.GRU(in_size, hidden_size, batch_first=True)
        self.dense1 = nn.Linear(hidden_size, 10)
        self.dense2 = nn.Linear(10, 1)

    def forward(self, sequence):
        gru_out, _ = self.gru(sequence)
        x = self.dense1(gru_out[:,-1,:])
        x = torch.relu(x)
        x = self.dense2(x)
        return x
