import numpy as np
import torch
from torch.autograd import Variable

from protein_transformer.protein.Sidechains import NUM_PREDICTED_ANGLES
from .Decoder import Decoder
from .Encoder import Encoder


class Transformer(torch.nn.Module):
    """
    Transformer based model.
    # TODO why are masks created in/outside of the transformer?
    """
    def __init__(self, dm, dff, din, dout, n_heads, n_enc_layers, n_dec_layers,
                 max_seq_len, pad_char, missing_coord_filler, device, dropout, fraction_complete_tf,
                 fraction_subseq_tf, angle_mean_path):
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
        self.missing_coord_filler = missing_coord_filler
        self.device = device
        self.fraction_subseq_tf = fraction_subseq_tf
        self.fraction_complete_tf = fraction_complete_tf
        self.angle_mean_path = angle_mean_path

        self.decoder_sos_char = -0.1

        self.encoder = Encoder(self.din, dm, dff, n_heads, n_enc_layers, max_seq_len, dropout)
        self.decoder = Decoder(self.dout, dm, dff, n_heads, n_dec_layers, max_seq_len, dropout)
        self.output_projection = torch.nn.Linear(dm, self.dout)
        self.tanh = torch.nn.Tanh()
        self._init_parameters()

    def forward_tf(self, enc_input, dec_input):
        """
        Forward method of Transformer that uses teacher forcing.
        Includes Encoder, Decoder, and output projection.
        """
        src_mask = (enc_input != self.pad_char).unsqueeze(-2)
        tgt_mask = (dec_input != self.pad_char).any(dim=-1).unsqueeze(-2) & self.subsequent_mask(dec_input.shape[1])
        enc_output = self.encoder(enc_input, src_mask)
        dec_output = self.decoder(dec_input, enc_output, tgt_mask, src_mask)
        logits = self.output_projection(dec_output)
        return self.tanh(logits)


    def forward(self, enc_input, dec_input):
        """
        Makes predictions using teacher forcing, if requested. Otherwise,
        uses sequential decoding.
        """
        # Replace nans with missing character
        dec_input[torch.isnan(dec_input)] = self.missing_coord_filler
        has_missing_residues = torch.isnan(dec_input).all(dim=-1).any().byte()

        # Decoder input only receives time steps SOS..t-1
        dec_input[:, 1:] = dec_input.clone()[:, :-1]
        dec_input[:, 0] = self.decoder_sos_char


        # Switch to the full teacher forcing function if requested and no missing residues
        if (not has_missing_residues) and \
                    (self.fraction_complete_tf == 1 or
                     self.fraction_subseq_tf == 1 or
                     np.random.random() < self.fraction_complete_tf):
            return self.forward_tf(enc_input, dec_input)

        # Otherwise, proceed with a method that will use sub-sequence level teacher forcing
        src_mask = (enc_input != self.pad_char).unsqueeze(-2)
        enc_output = self.encoder(enc_input, src_mask)
        max_len = enc_input.shape[1]

        # Construct a placeholder for the predictions, starting w/the true angles (augmented with an SOS char)
        working_input_seq = dec_input.clone()

        for t in range(1, max_len):
            # Slice the relevant subset of the output to provide as input. t == 1 : SOS, else: decoder output
            dec_input = Variable(working_input_seq.data[:, :t])
            tgt_mask = (dec_input != self.pad_char).any(dim=-1).unsqueeze(-2) & self.subsequent_mask(dec_input.shape[1])

            # Using the output so far (dec_input), run the decoder one step
            dec_output = self.decoder(dec_input, enc_output, tgt_mask, src_mask)
            angles = self.output_projection(dec_output[:, -1])
            angles = self.tanh(angles)

            # Update the next timestep in the placeholder with predicted angle randomly or if next residue is missing
            feed_prediction = t + 1 < max_len and ((np.random.random() > self.fraction_subseq_tf) or
                                                   dec_input[:, t].all(dim=-1) == self.missing_coord_filler)
            if t + 1 < max_len and feed_prediction:
                working_input_seq.data[:, t] = angles.data

        return self.tanh(self.output_projection(dec_output))


    def _init_parameters(self):
        """
        Initialize model parameters. Also, attempt to initialize model output to
        angle means.
        """
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        # Initialize final projection layer to predict mean of angle distribution
        angle_means = np.load(self.angle_mean_path)
        self.output_projection.bias = torch.nn.Parameter(torch.FloatTensor(np.arctanh(angle_means)))
        torch.nn.init.xavier_uniform_(self.output_projection.weight, gain=0.00001)


    def subsequent_mask(self, length):
        """
        Returns a mask such that for position i, all positions i+1 ... dim are masked.
        """
        shape = (1, length, length)
        mask = 1 - np.triu(np.ones(shape), k=1)
        return torch.from_numpy(mask).bool().to(self.device)


    def predict(self, enc_input):
        """
        Makes predictions with self-recursive decoding.
        """
        src_mask = (enc_input != self.pad_char).unsqueeze(-2)
        enc_output = self.encoder(enc_input, src_mask)
        max_len = enc_input.shape[1]

        # Construct a placeholder for the predicted data, starting with a special value of SOS
        working_input_seq = Variable(
            torch.zeros((enc_input.shape[0], max_len - 1, NUM_PREDICTED_ANGLES * 2), device=self.device,
                       requires_grad=True))
        working_input_seq[:, 0] = enc_input[:,0]  # Add SOS character to working decoder input / output

        for t in range(1, max_len):
            # Slice the relevant subset of the output to provide as input. t == 1 : SOS, else: decoder output
            dec_input = Variable(working_input_seq.data[:, :t])
            tgt_mask = (dec_input != self.pad_char).unsqueeze(
                -2) & self.subsequent_mask(dec_input.shape[1])

            # Using the output so far (dec_input), run the decoder one step
            dec_output = self.decoder(dec_input, enc_output, tgt_mask, src_mask)
            angles = self.output_projection(dec_output[:, -1])
            angles = self.tanh(angles)

            # Update the next timestep in the placeholder with predicted angle randomly or if the next res is missing
            if t + 1 < max_len:
                working_input_seq.data[:, t] = angles.data

        return self.tanh(self.output_projection(dec_output))
