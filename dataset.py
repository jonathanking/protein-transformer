import numpy as np
import torch
import torch.utils.data

from protein.Sidechains import NUM_PREDICTED_ANGLES, NUM_PREDICTED_COORDS
from models.transformer.Models import SOS_CHAR
MAXDATALEN = 500


def paired_collate_fn(insts):
    """
    This function creates mini-batches (4-tuples) of src_seq/pos,
    trg_seq/pos Tensors. insts is a list of tuples, each containing one src
    and one target seq.
    """
    sequences, angles, coords = list(zip(*insts))
    sequences = collate_fn(sequences, pad_dim=20)
    angles = collate_fn(angles, pad_dim=NUM_PREDICTED_ANGLES * 2)
    coords = collate_fn(coords, pad_dim=3, coords=True)
    return (*sequences, *angles, *coords)


def collate_fn(insts, pad_dim, coords=False):
    """
    Pad the instance to the max seq length in batch.
    """
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
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.
    """
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
    """
    Add a special 'start of sentence' character to each sequence.
    """
    start = np.asarray([sos_char] * two_dim_array.shape[1]).reshape(1, two_dim_array.shape[1])
    return np.concatenate([start, two_dim_array])


def prepare_dataloaders(data, args, use_start_char=False):
    """
    Using the pre-processed data, stored in a nested Python dictionary, this
    function returns train, validation, and test set dataloaders with 2 workers
    each. There are multiple validation sets in ProteinNet. Currently, this
    method only returns set '70'.
    """
    # TODO: load and evaluate multiple validation sets
    train_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['train']['seq']*args.repeat_train,
            crds=data['train']['crd']*args.repeat_train,
            angs=data['train']['ang']*args.repeat_train,
            add_start_character_to_input=use_start_char),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['valid'][70]['seq'],
            crds=data['valid'][70]['crd'],
            angs=data['valid'][70]['ang'],
            add_start_character_to_input=use_start_char),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn)

    test_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['test']['seq'],
            crds=data['test']['crd'],
            angs=data['test']['ang'],
            add_start_character_to_input=use_start_char),
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader, test_loader