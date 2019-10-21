""" This file contains models based off of ones I built in a NMT course. """
import torch
import torch.nn as nn
from models.transformer_nmt.Encoder import Encoder
import numpy as np
from protein.Sidechains import NUM_PREDICTED_ANGLES

PAD_CHAR = 0
NUM_RESIDUES = 0

class EncoderOnlyTransformer(nn.Module):
    """ A Transformer that only uses Encoder layers. """

    def __init__( self, nlayers, nhead, dmodel, dff, max_seq_len, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(20, dmodel, dff, nhead, nlayers, max_seq_len, dropout)
        self.output_projection = torch.nn.Linear(dmodel, NUM_PREDICTED_ANGLES*2)
        self.tanh = nn.Tanh()
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.output_projection.bias = nn.Parameter(torch.FloatTensor(np.arctanh(self.load_angle_means(\
            "data/proteinnet/casp12_190927_100_mean.npy"))))
        nn.init.xavier_normal_(self.output_projection.weight, gain=0.00001)

    def forward(self, enc_input):
        src_mask = (enc_input != PAD_CHAR).unsqueeze(-2)
        enc_output = self.encoder(enc_input, src_mask)
        enc_output = self.output_projection(enc_output)
        enc_output = self.tanh(enc_output)
        return enc_output
