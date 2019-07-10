import numpy as np
import torch
import torch.utils.data

from protein.Sidechains import NUM_PREDICTED_ANGLES


def paired_collate_fn(insts):
    """ This function creates mini-batches (4-tuples) of src_seq/pos,
        trg_seq/pos Tensors. insts is a list of tuples, each containing one src
        and one target seq.
        """
    sequences, angles = list(zip(*insts))
    sequences = collate_fn(sequences, pad_dim=20)
    angles = collate_fn(angles, pad_dim=NUM_PREDICTED_ANGLES * 2)
    return (*sequences, *angles)


def collate_fn(insts, pad_dim):
    """ Pad the instance to the max seq length in batch """

    max_len = max(len(inst) for inst in insts)
    batch_seq = []
    for inst in insts:
        z = np.zeros((max_len - len(inst), pad_dim))
        c = np.concatenate((inst, z), axis=0)
        batch_seq.append(c)
    batch_seq = np.array(batch_seq)

    batch_pos = np.array([
        [pos_i+1 if w_i.any() else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])  # position arr

    batch_seq = torch.FloatTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, seqs=None, angs=None, crds=None):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) )

        self._seqs = seqs
        self._angs = angs

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._seqs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._angs is not None:
            return self._seqs[idx], self._angs[idx]
        return self._seqs[idx]
