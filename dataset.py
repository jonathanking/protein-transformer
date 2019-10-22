import numpy as np
import torch
import torch.utils.data

from protein.Sidechains import NUM_PREDICTED_ANGLES, NUM_PREDICTED_COORDS
from models.transformer.Models import SOS_CHAR
MAXDATALEN = 500
# TODO inspect max size - is 500 appropriate?


def paired_collate_fn(insts):
    """ This function creates mini-batches (4-tuples) of src_seq/pos,
        trg_seq/pos Tensors. insts is a list of tuples, each containing one src
        and one target seq.
        """
    sequences, angles, coords = list(zip(*insts))
    sequences = collate_fn(sequences, pad_dim=20)
    angles = collate_fn(angles, pad_dim=NUM_PREDICTED_ANGLES * 2)
    coords = collate_fn(coords, pad_dim=3, coords=True)
    return (*sequences, *angles, *coords)


def paired_collate_fn_with_len(insts):
    """ This function creates mini-batches (4-tuples) of src_seq/pos,
        trg_seq/pos Tensors. insts is a list of tuples, each containing one src
        and one target seq.
        """
    sequences, angles, coords = list(zip(*insts))
    seq_seq, seq_pos = collate_fn(sequences, pad_dim=20)
    ang_seq, ang_pos = collate_fn(angles, pad_dim=NUM_PREDICTED_ANGLES * 2)
    crd_seq, crd_pos = collate_fn(coords, pad_dim=3, coords=True)
    # Compute sorted lengths
    lens = torch.LongTensor([min(MAXDATALEN, len(s)) for s in sequences])
    sorted_lengths, indices = torch.sort(lens.view(-1), dim=0, descending=True)
    # Sort data by lengths, as needed for an RNN and pack_padded_sequence
    seq_seq, seq_pos = seq_seq[indices], seq_pos[indices]
    ang_seq, ang_pos = ang_seq[indices][:, :sorted_lengths[0]], ang_pos[indices][:, :sorted_lengths[0]]
    crd_seq, crd_pos = crd_seq[indices][:, :sorted_lengths[0]*NUM_PREDICTED_COORDS], crd_pos[indices][:, :sorted_lengths[0]*NUM_PREDICTED_COORDS]
    return sorted_lengths, seq_seq, seq_pos, ang_seq, ang_pos, crd_seq, crd_pos


def collate_fn(insts, pad_dim, coords=False):
    """ Pad the instance to the max seq length in batch """

    max_len = max(len(inst) for inst in insts)
    batch_seq = []
    for inst in insts:
        z = np.zeros((max_len - len(inst), pad_dim))
        c = np.concatenate((inst, z), axis=0)
        batch_seq.append(c)
    batch_seq = np.array(batch_seq)
    if coords:
        batch_seq = batch_seq[:,:MAXDATALEN*NUM_PREDICTED_COORDS]
    else:
        batch_seq = batch_seq[:,:MAXDATALEN]

    batch_pos = np.array([
        [pos_i+1 if w_i.any() else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])  # position arr

    batch_seq = torch.FloatTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, seqs=None, angs=None, crds=None, add_start_character_to_input=False):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))

        # We must add "start of sentence" characters to the sequences that are fed to the Transformer.
        # This enables us to input time t and expect the model to predict time t+1.
        # Coordinates do not need this requirement since they are not directly input to the transformer.
        if add_start_character_to_input:
            self._seqs = [add_start_char(s) for s in seqs]
            self._angs = [add_start_char(a) for a in angs]
            self._crds = crds
        else:
            self._seqs = seqs
            self._angs = angs
            self._crds = crds

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._seqs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._angs is not None:
            return self._seqs[idx], self._angs[idx], self._crds[idx]
        return self._seqs[idx]


def add_start_char(two_dim_array, sos_char=SOS_CHAR):
    """ Add a special 'start of sentence' character to each sequence. """
    start = np.asarray([sos_char] * two_dim_array.shape[1]).reshape(1, two_dim_array.shape[1])
    return np.concatenate([start, two_dim_array])
