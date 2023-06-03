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
                 device: str):
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

    def forward(self, x: torch.Tensor):
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        return self.label(final_hidden_state[-1])
