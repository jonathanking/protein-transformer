import torch
from torch import nn
from torch.autograd import Variable

from protein.Sidechains import NUM_PREDICTED_ANGLES, NUM_PREDICTED_COORDS


class MyRNN(nn.Module):
    def __init__(self, H, D_in=20, D_out=NUM_PREDICTED_ANGLES, num_layers=1, bidirectional=True):
        #TODO implement multiple layers
        super(MyRNN, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden2out = nn.Linear(self.num_direction * self.hidden_dim, D_out *2)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)))
        return h, c

    def forward(self, sequence, lengths, h, c):
        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)
        output = output.view(output.shape[0], output.shape[1], 24)
        return output
