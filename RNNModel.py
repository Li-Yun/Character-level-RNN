import torch
from torch import nn
import torch.nn.functional as F


class CharacterRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {val: key for key, val in self.int2char.items()}

        # define LSTM layers
        self.lstm = nn.LSTM(len(self.chars), self.n_hidden, self.n_layers,
                            dropout=self.drop_prob, batch_first=True)
        # define a dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        # define a fully-connected layer
        self.fc = nn.Linear(self.n_hidden, len(self.chars))

    def init_hidden(self, batch_size, train_on_gpu):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

    def forward(self, inputs, hidden):
        """
        Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`.
        """
        # get the outputs and the new hidden state from the lstm
        r_out, hidden = self.lstm(inputs, hidden)
        out = self.dropout(r_out)
        # Stack up LSTM outputs
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

