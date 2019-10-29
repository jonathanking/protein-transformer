import torch
from .Encoder import Encoder
from .Decoder import Decoder
from protein.Sidechains import NUM_PREDICTED_ANGLES
from torch.autograd import Variable



import numpy as np

class Transformer(torch.nn.Module):
    """
    Transformer based model.
    """
    def __init__(self, dm, dff, din, dout, n_heads, n_enc_layers, n_dec_layers, max_seq_len, pad_char, device, dropout=0.1):
        super(Transformer, self).__init__()
        self.din = din
        self.dout = dout
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_laers = n_dec_layers
        self.max_seq_len = max_seq_len
        self.pad_char = pad_char
        self.device = device

        self.encoder = Encoder(self.din, dm, dff, n_heads, n_enc_layers, max_seq_len, dropout)
        self.decoder = Decoder(self.dout, dm, dff, n_heads, n_dec_layers, max_seq_len, dropout)
        self.output_projection = torch.nn.Linear(dm, self.dout)
        self._init_parameters()

    def forward(self, enc_input, dec_input):
        # TODO why are masks created in/outside of the transformer?
        src_mask = (enc_input != self.pad_char).unsqueeze(-2)
        tgt_mask = (dec_input != self.pad_char).unsqueeze(-2) & self.subsequent_mask(dec_input.shape[1])
        enc_output = self.encoder(enc_input, src_mask)
        dec_output = self.decoder(dec_input, enc_output, tgt_mask, src_mask)
        logits = self.output_projection(dec_output)
        return logits

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def subsequent_mask(self, length):
        """
        Returns a mask such that for position i, all positions i+1 ... dim are masked.
        """
        shape = (1, length, length)
        mask = 1 - np.triu(np.ones(shape), k=1)
        return torch.from_numpy(mask).bool().to(self.device)


    def predict(self, src_seq, src_pos):
        """
        Uses the model to make a prediction/do inference given only the src_seq. This is in contrast to training
        when the model is allowed to make use of the tgt_seq for teacher forcing.
        """
        src_seq = self.input_embedding(src_seq)
        enc_output, *_ = self.encoder(src_seq, src_pos)
        max_len = src_seq.shape[1]

        # Construct a placeholder for the data, starting with a special value of SOS
        working_input_seq = Variable(torch.ones((src_seq.shape[0], max_len-1, NUM_PREDICTED_ANGLES*2),
                                                device=self.device, requires_grad=True) * SOS_CHAR)

        for t in range(1, max_len):
            # Slice the relevant subset of the output to provide as input. t == 1 : SOS, else: decoder output
            dec_input = Variable(working_input_seq.data[:, :t])
            dec_input_pos = Variable(src_pos[:, :t])

            # Embed the output so far into the decoder's input space, and run the decoder one step
            dec_input = self.tgt_embedding(dec_input)
            dec_output, *_ = self.decoder(dec_input, dec_input_pos, src_seq, enc_output)
            angles = self.tgt_angle_prj(dec_output[:,-1])
            angles = self.tanh(angles)

            # Update our placeholder with the predicted angles thus far
            if t+1 < max_len:
                working_input_seq.data[:, t] = angles.data

        return self.tanh(self.tgt_angle_prj(dec_output))

