import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from typing import Union


class LSTMForcaster(nn.Module):
    def __init__(self,
                     embedding_length: int,
                     hidden_size: int,
                     output_size: int,
                     lstm_layers: int,
                     device: str
                     ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.input_size = embedding_length
        self.device = device
        self.lstm = nn.LSTM(embedding_length,
                            hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.label = nn.Linear(hidden_size, output_size)


    def forward(self, x:torch.Tensor):
        # Hidden state and cell state
        h_0 = Variable(torch.zeros(self.lstm_layers, x.shape[0], self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.lstm_layers, x.shape[0], self.hidden_size).cuda())

        # move hidden state to device
        if self.device == "cuda":
            h_0 = h_0.to(self.device)
            c_0 = c_0.to(self.device)

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        return self.label(final_hidden_state[-1]) 
