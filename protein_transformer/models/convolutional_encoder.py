""" This model behaves similarlry to other Transformer based models, but incorporates
    several layers of 1D sequence convolutions after embedding and before attention.
"""
import numpy as np
import torch
import torch.nn as nn

from protein_transformer.models.transformer.Sublayers import Embeddings, PositionalEncoding
from protein_transformer.protein.Structure import NUM_PREDICTED_ANGLES
from protein_transformer.models.transformer.Encoder import Encoder, EncoderLayer


class ConvEncoderOnlyTransformer(nn.Module):
    """ A Transformer that starts with 1D sequence convolutions before applying attention. """

    def __init__( self, nlayers, nhead, dmodel, dff, max_seq_len, vocab, angle_means, use_tanh_out, conv_kernel_sizes,
                  conv_dim_reductions, use_embedding, dropout=0.1):
        super().__init__()
        self.angle_means = angle_means
        self.vocab = vocab
        self.encoder = ConvolutionalEncoder(len(vocab), dmodel, dff, nhead, nlayers, max_seq_len, dropout,
                                            conv_kernel_sizes, conv_dim_reductions, use_embedding)
        self.output_projection = torch.nn.Linear(self.encoder.conv_out_size(), NUM_PREDICTED_ANGLES*2)
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


class ConvolutionalEncoder(torch.nn.Module):
    """
    Transformer encoder model that starts with sequence convolutions.
    """

    def __init__(self, din, dm, dff, n_heads, n_enc_layers, max_seq_len, dropout, conv_kernel_sizes,
                 conv_dim_reductions, use_embedding):
        super(ConvolutionalEncoder, self).__init__()
        self.din = din
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.max_seq_len = max_seq_len
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_dim_reductions = conv_dim_reductions
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.emb_dropout = torch.nn.Dropout(dropout)
            self.input_embedding = Embeddings(self.din, self.dm)
            self.positional_enc = PositionalEncoding(dm, dropout, max_seq_len)
        else:
            self.positional_enc = PositionalEncoding(din, dropout, max_seq_len)

        self.conv_layers = torch.nn.ModuleList(self.make_sequence_conv_layers(conv_kernel_sizes, conv_dim_reductions))

        self.enc_layers = torch.nn.ModuleList([EncoderLayer(self.conv_out_size(), dff, n_heads, dropout)
                                               for _ in range(self.n_enc_layers)])

    def conv_out_size(self):
        d = self.dm if self.use_embedding else self.din
        for dr in self.conv_dim_reductions:
            d /= dr
        return int(d)

    def make_sequence_conv_layers(self, conv_kernel_sizes, conv_dim_reductions):
        conv_layers = []
        din = self.dm if self.use_embedding else self.din

        for k, dim_red in zip(conv_kernel_sizes, conv_dim_reductions):
            dout = int(din // dim_red)
            c = self.make_length_preserving_conv_layer(k, din, dout)
            conv_layers.append(c)
            din = dout
        return conv_layers

    def forward(self, src_seq, src_mask):
        if self.use_embedding:
            enc_output = self.input_embedding(src_seq)
            enc_output = self.emb_dropout(enc_output + self.positional_enc(enc_output))
        else:
            enc_output = torch.nn.functional.one_hot(src_seq, num_classes=self.din).float()
        enc_output = enc_output.transpose(-1, -2)

        for conv_layer in self.conv_layers:
            enc_output = conv_layer(enc_output)

        enc_output = enc_output.transpose(-1, -2)
        if not self.use_embedding:
            enc_output += self.positional_enc(enc_output)

        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, src_mask)
        return enc_output

    @staticmethod
    def make_length_preserving_conv_layer(kernel_size, d_in, d_out):
        assert kernel_size % 2 != 0, "Kernel size must be odd to maintain sequence length."
        padding = int((kernel_size - 1) / 2)
        return torch.nn.Conv1d(d_in, d_out, kernel_size, padding=padding)

if __name__ == '__main__':
    m = ConvolutionalEncoder(20,512,1024,2,6,45,0.1,[3,7,11],[2,2,2])
    seq = torch.randint(0,20,(1,45))
    mask = (seq != 0).unsqueeze(-2)
    x = m(seq, mask)
    print(x)