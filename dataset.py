import numpy as np
import torch
import torch.utils.data

from transformer import Constants

def paired_collate_fn(insts):
    """ This function creates mini-batches (4-tuples) of src_seq/pos,
        trg_seq/pos Tensors. insts is a list of tuples, each containing one src
        and one target seq.
        """
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    # TODO: How to collate input/output such that Tensor datatypes are correct
    # TODO: Must create separate iterators for each input/output data

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts]) # pads as many chars as necessary for each inst

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq]) # position arr

    batch_seq = torch.LongTensor(batch_seq) 
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(
        self, seqs=None, angs=None):

        assert seqs
        assert not angs or (len(seqs) == len(angs))

        self._seqs = seqs
        self._angs = angs

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._seqs)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._angs:
            return self._seqs[idx], self._angs[idx]
        return self._seqs[idx]
