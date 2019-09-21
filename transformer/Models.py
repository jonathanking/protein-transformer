''' Define the Transformer model '''
import os.path as path

import numpy as np
import torch
import torch.nn as nn

from transformer.Layers import EncoderLayer, DecoderLayer
from protein.Sidechains import NUM_PREDICTED_ANGLES

__author__ = "Yu-Hsiang Huang"


def get_non_pad_mask(seq):
    assert seq.dim() == 3
    # return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
    return (seq != 0).any(dim=-1).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    # TODO add ability to use np.nan within model, i.e. padding_mask = torch.isnan(seq_k).all(dim=-1)
    padding_mask = (seq_k == 0).all(dim=-1)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s, dim_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, postnorm, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, postnorm=postnorm)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        # TODO mofidy masks to allow for np.nan
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = src_seq + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output,
                                                 non_pad_mask=non_pad_mask,
                                                 slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, postnorm, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position, d_model,
                                                                                     padding_idx=0),
                                                         freeze=True)

        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, postnorm=postnorm)
                                          for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = tgt_seq + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output,
                                                               enc_output,
                                                               non_pad_mask=non_pad_mask,
                                                               slf_attn_mask=slf_attn_mask,
                                                               dec_enc_attn_mask=dec_enc_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, args, d_angle=NUM_PREDICTED_ANGLES * 2,
                 d_word_vec=20, d_model=512,
                 d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()
        init_w_angle_means, data_path, len_max_seq = not args.without_angle_means, args.data, args.max_token_seq_len
        self.encoder = Encoder(len_max_seq=len_max_seq,
                               d_word_vec=d_word_vec,
                               d_model=d_model,
                               d_inner=d_inner,
                               n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_k,
                               d_v=d_v,
                               postnorm=args.postnorm,
                               dropout=dropout)

        self.decoder = Decoder(len_max_seq=len_max_seq,
                               d_model=d_model,
                               d_inner=d_inner,
                               n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_k,
                               d_v=d_v,
                               postnorm=args.postnorm,
                               dropout=dropout)

        self.input_embedding = nn.Linear(d_word_vec, d_model)  # nn.Embedding(d_word_vec, d_angle)
        self.tgt_angle_prj = nn.Linear(d_model, d_angle)
        self.tgt_embedding = nn.Linear(d_angle, d_model)
        nn.init.xavier_normal_(self.tgt_angle_prj.weight)
        nn.init.xavier_normal_(self.input_embedding.weight)
        nn.init.xavier_normal_(self.tgt_embedding.weight)

        # Initialize output linear layer bias with angle means
        if init_w_angle_means:
            self.tgt_angle_prj.bias = nn.Parameter(torch.FloatTensor(np.arctanh(self.load_angle_means(data_path))))
            # TODO is it better to init with nn.init.zeros_(self.tgt_angle_prj.weight) ?
            nn.init.xavier_normal_(self.tgt_angle_prj.weight, gain=0.00001)
        self.tanh = nn.Tanh()

    def load_angle_means(self, data_path):
        """
        Loads the average angle vector in order to initialize the bias out the output layer. This allows the model to
        begin predicting the average angle vectors and must only learn to predict the difference.
        """
        data, ext = path.splitext(data_path)
        angle_mean_path = data + "_mean.npy"
        if not path.exists(angle_mean_path):
            angle_mean_path_new = "protein/190602_query4_mean.npy"
            print(f"[Info] Unable to find {angle_mean_path}. Loading angle means from {angle_mean_path_new} instead.")
            angle_mean_path = angle_mean_path_new
        angle_means = np.load(angle_mean_path)
        return angle_means

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        src_seq = self.input_embedding(src_seq)
        tgt_seq = self.tgt_embedding(tgt_seq)
        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        angles = self.tgt_angle_prj(dec_output)
        angles = self.tanh(angles)
        return angles

