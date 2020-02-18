""" Represents a base model for predicting protein structure. """

import torch.nn as nn
import torch
import numpy as np

from protein_transformer.protein.Structure import NUM_PREDICTED_ANGLES


class BaseModel(nn.Module):
    """ The Base model. """

    def __init__( self, nlayers, nhead, dmodel, dff, max_seq_len, vocab, angle_means, dropout=0.1):
        super().__init__()
        self.angle_means = angle_means
        self.vocab = vocab
        self.encoder = Encoder(len(vocab), dmodel, dff, nhead, nlayers, max_seq_len, dropout)
        self.output_projection = torch.nn.Linear(dmodel, NUM_PREDICTED_ANGLES*2)
        self.tanh = nn.Tanh()
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.output_projection.bias = nn.Parameter(torch.tensor(np.arctanh(self.angle_means), dtype=torch.float32))
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, enc_input, dec_input=None):
        src_mask = (enc_input != self.vocab.pad_id).unsqueeze(-2)
        enc_output = self.encoder(enc_input, src_mask)
        enc_output = self.output_projection(enc_output)
        enc_output = self.tanh(enc_output)
        return enc_output

    def predict(self, enc_input):
        return self.forward(enc_input)