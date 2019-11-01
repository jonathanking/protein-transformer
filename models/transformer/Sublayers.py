import torch
import numpy as np


class SublayerConnection(torch.nn.Module):
    """
    Does residual + layer norm of input. Modular design inspired from Harvard
    NLP.
    http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder-and-decoder-stacks
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, layer_input, layer):
        return layer_input + self.dropout(layer(self.norm(layer_input)))


class PositionwiseFeedForward(torch.nn.Module):
    """
    Position-wise Feed Forward network sublayer for the Transformer model.
    """
    def __init__(self, dm, dh, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.dm = dm
        self.dh = dh
        self.layer1 = torch.nn.Linear(dm, dh)
        self.layer2 = torch.nn.Linear(dh, dm)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_seq):
        return self.layer2(self.dropout(self.relu(self.layer1(input_seq))))


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding layer for the Transformer model.
    From Alexander Rush,
    https://github.com/harvardnlp/annotated-transformer/blob/master/The%20Annotated%20Transformer.ipynb
    """

    def __init__(self, dm, dropout, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, dm)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., dm, 2) *
                             -(np.log(10000.0) / dm))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":
    seq = torch.ones(8, 7, 64)
    penc = PositionalEncoding(64, 300)
    print(penc(seq).shape)
    print(penc(seq))
