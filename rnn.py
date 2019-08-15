import os.path as path

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from protein.Sidechains import NUM_PREDICTED_ANGLES, NUM_PREDICTED_COORDS


class MyRNN(nn.Module):
    def __init__(self, args, H, D_in=20, D_out=NUM_PREDICTED_ANGLES, num_layers=1, bidirectional=True, device=torch.device('cuda')):
        super(MyRNN, self).__init__()
        self.hidden_dim = H
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(D_in, H, num_layers, bidirectional=bidirectional, batch_first=True)
        self.hidden2out = nn.Linear(self.num_direction * self.hidden_dim, D_out *2)
        self.device = device
        self.d_out = D_out
        self.data_path = args.data
        if not args.without_angle_means:
            self.hidden2out.bias = nn.Parameter(torch.FloatTensor(np.arctanh(self.load_angle_means())))
            # TODO is it better to init with nn.init.zeros_(self.tgt_angle_prj.weight) ?
            nn.init.xavier_normal_(self.hidden2out.weight)

    def init_hidden(self, batch_size):
        """ Initialize the hidden state vectors at the start of a batch iteration. """
        h, c = (Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)).to(self.device),
                Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)).to(self.device))
        return h, c

    def forward(self, sequence, lengths, h, c):
        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)
        output = output.view(output.shape[0], output.shape[1], self.d_out * 2)
        return output

    def load_angle_means(self):
        """
        Loads the average angle vector in order to initialize the bias out the output layer. This allows the model to
        begin predicting the average angle vectors and must only learn to predict the difference.
        """
        data, ext = path.splitext(self.data_path)
        angle_mean_path = data + "_mean.npy"
        if not path.exists(angle_mean_path):
            angle_mean_path_new = "protein/190602_query4_mean.npy"
            print(f"[Info] Unable to find {angle_mean_path}. Loading angle means from {angle_mean_path_new} instead.")
            angle_mean_path = angle_mean_path_new
        angle_means = np.load(angle_mean_path)
        return angle_means
