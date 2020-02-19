""" This file contains models based off of ones I built in a NMT course. """
import numpy as np
import torch
import torch.nn as nn

from protein_transformer.protein.Structure import NUM_PREDICTED_ANGLES
from .transformer.Encoder import Encoder


class EncoderOnlyTransformer(nn.Module):
    """ A Transformer that only uses Encoder layers. """

    def __init__( self, nlayers, nhead, dmodel, dff, max_seq_len, vocab, angle_means, use_tanh_out, dropout=0.1):
        super().__init__()
        self.angle_means = angle_means
        self.vocab = vocab
        self.encoder = Encoder(len(vocab), dmodel, dff, nhead, nlayers, max_seq_len, dropout)
        self.output_projection = torch.nn.Linear(dmodel, NUM_PREDICTED_ANGLES*2)
        self.use_tanh_out = use_tanh_out
        if use_tanh_out:
            self.tanh = nn.Tanh()
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.use_tanh_out:
            am = np.arctanh(self.angle_means)
        else:
            am = self.angle_means
        self.output_projection.bias = nn.Parameter(torch.tensor(am, dtype=torch.float32))

        nn.init.zeros_(self.output_projection.weight)

    def forward(self, enc_input, dec_input=None):
        src_mask = (enc_input != self.vocab.pad_id).unsqueeze(-2)
        enc_output = self.encoder(enc_input, src_mask)
        enc_output = self.output_projection(enc_output)
        if self.use_tanh_out:
            enc_output = self.tanh(enc_output)
        return enc_output

    def predict(self, enc_input):
        return self.forward(enc_input)
