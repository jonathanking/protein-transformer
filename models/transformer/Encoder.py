import torch
from .Sublayers import PositionwiseFeedForward, PositionalEncoding, SublayerConnection
from .Attention import MultiHeadedAttention

class Encoder(torch.nn.Module):
    """
    Transformer encoder model.
    """

    def __init__(self, din, dm, dff, n_heads, n_enc_layers, max_seq_len, dropout):
        super(Encoder, self).__init__()
        self.din = din
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.max_seq_len = max_seq_len

        self.emb_dropout = torch.nn.Dropout(dropout)
        self.input_embedding = torch.nn.Embedding(self.din, self.dm)
        self.positional_enc = PositionalEncoding(dm, dropout, max_seq_len)

        self.enc_layers = torch.nn.ModuleList([EncoderLayer(dm, dff, n_heads, dropout) for _ in range(self.n_enc_layers)])

    def forward(self, src_seq, src_mask):
        enc_output = self.input_embedding(src_seq)
        enc_output = self.emb_dropout(enc_output + self.positional_enc(enc_output))
        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output, src_mask)
        return enc_output


class EncoderLayer(torch.nn.Module):
    """
    Transformer encoder layer.
    """

    def __init__(self, dm, dff, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads

        self.self_attn = MultiHeadedAttention(dm, n_heads)
        self.pwff = PositionwiseFeedForward(dm, dff, dropout)
        self.sublayer_connections = torch.nn.ModuleList([SublayerConnection(dm, dropout) for _ in range(2)])

    def forward(self, enc_input, enc_input_mask):
        enc_output = self.sublayer_connections[0](enc_input, lambda x: self.self_attn(x, x, x, mask=enc_input_mask))
        enc_output = self.sublayer_connections[1](enc_output, self.pwff)
        return enc_output