import numpy as np
import torch
import torch.utils.data

from protein.Sidechains import NUM_PREDICTED_COORDS

def make_paired_collate_fn_with_max_len(max_seq_len):

    def paired_collate_fn(insts):
        """
        This function creates mini-batches (3-tuples) of sequence, angle and
        coordinate Tensors. insts is a list of tuples, each containing one src
        seq, and target angles and coordindates.
        """
        sequences, angles, coords = list(zip(*insts))
        sequences = collate_fn(sequences, sequences=True, max_seq_len=max_seq_len)
        angles = collate_fn(angles, max_seq_len=max_seq_len)
        coords = collate_fn(coords, coords=True, max_seq_len=max_seq_len)
        return sequences, angles, coords

    return paired_collate_fn


def collate_fn(insts, coords=False, sequences=False, max_seq_len=None):
    """
    Given a list of tuples to be stitched together into a batch, this function
    pads each instance to the max seq length in batch and returns a batch
    Tensor.
    """
    max_batch_len = max(len(inst) for inst in insts)
    batch = []
    for inst in insts:
        if sequences:
            z = np.ones((max_batch_len - len(inst))) * VOCAB.pad_id
        else:
            z = np.zeros((max_batch_len - len(inst), inst.shape[-1]))
        c = np.concatenate((inst, z), axis=0)
        batch.append(c)
    batch = np.array(batch)

    # Trim batch to be less than maximum length
    if coords:
        batch = batch[:,:max_seq_len*NUM_PREDICTED_COORDS]
    else:
        batch = batch[:,:max_seq_len]

    if sequences:
        batch = torch.LongTensor(batch)
    else:
        batch = torch.FloatTensor(batch)

    return batch



class ProteinVocabulary(object):
    """
    Represents the 'vocabulary' of amino acids for encoding a protein sequence.
    Includes pad, sos, sos, and unknown characters as well as the 20 standard
    amino acids.
    """
    def __init__(self):
        self.aa2id = dict()
        self.pad_id = 0         # Pad character
        self.sos_id = 1         # SOS character
        self.eos_id = 2         # EOS character
        self.unk_id = 3         # unknown character
        self.aa2id['_'] = self.pad_id
        self.aa2id['<'] = self.sos_id
        self.aa2id['>'] = self.eos_id
        self.aa2id['?'] = self.unk_id
        self._id2aa = {v: k for k, v in self.aa2id.items()}
        self.stdaas = "ARNDCQEGHILKMFPSTWYV"
        for aa in self.stdaas:
            self.add(aa)

    def __getitem__(self, aa):
        return self.aa2id.get(aa, self.unk_id)

    def __contains__(self, aa):
        return aa in self.aa2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.aa2id)

    def __repr__(self):
        return f"ProteinVocabulary[size={len(self)}]"

    def id2aa(self, id):
        return self._id2aa[id]

    def add(self, aa):
        if aa not in self:
            aaid = self.aa2id[aa] = len(self)
            self._id2aa[aaid] = aa
            return aaid
        else:
            return self[aa]

    def aa_seq2indices(self, seq, add_sos_eos=True):
        if add_sos_eos:
            return [self["<"]] + [self[aa] for aa in seq] + [self[">"]]
        else:
            return [self[aa] for aa in seq]

    def indices2aa_seq(self, indices, include_sos_eos=False):
        seq = ""
        for i in indices:
            c = self.id2aa(i)
            if include_sos_eos or (i != self.sos_id and i != self.eos_id and i != self.pad_id):
                seq += c
        return seq



class ProteinDataset(torch.utils.data.Dataset):
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.
    """
    def __init__(self, seqs=None, angs=None, crds=None, add_sos_eos=True):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))
        # TODO use raw sequences in dataset; allows for pad character
        self.vocab = ProteinVocabulary()
        self._seqs = [VOCAB.aa_seq2indices(s, add_sos_eos) for s in seqs]
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


def prepare_dataloaders(data, args, max_seq_len, num_workers=2):
    """
    Using the pre-processed data, stored in a nested Python dictionary, this
    function returns train, validation, and test set dataloaders with 2 workers
    each. There are multiple validation sets in ProteinNet. Currently, this
    method only returns set '70'.
    """
    collate = make_paired_collate_fn_with_max_len(max_seq_len)

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    # TODO: load and evaluate multiple validation sets
    train_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['train']['seq']*args.repeat_train,
            crds=data['train']['crd']*args.repeat_train,
            angs=data['train']['ang']*args.repeat_train,
            add_sos_eos=args.add_sos_eos),
        num_workers=num_workers,
        batch_size=args.batch_size,
        collate_fn=collate,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['valid'][70]['seq'],
            crds=data['valid'][70]['crd'],
            angs=data['valid'][70]['ang'],
            add_sos_eos=args.add_sos_eos),
        num_workers=num_workers,
        batch_size=args.batch_size,
        collate_fn=collate,
        worker_init_fn=_init_fn)

    test_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['test']['seq'],
            crds=data['test']['crd'],
            angs=data['test']['ang'],
            add_sos_eos=args.add_sos_eos),
        num_workers=num_workers,
        batch_size=args.batch_size,
        collate_fn=collate,
        worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader

VOCAB = ProteinVocabulary()
