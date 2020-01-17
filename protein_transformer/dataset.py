import numpy as np
import torch
import torch.utils.data

from .protein.Sidechains import NUM_PREDICTED_COORDS
VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]
MAX_SEQ_LEN = 500


def paired_collate_fn(insts):
    """
    This function creates mini-batches (3-tuples) of sequence, angle and
    coordinate Tensors. insts is a list of tuples, each containing one src
    seq, and target angles and coordindates.
    """
    sequences, angles, coords = list(zip(*insts))
    sequences = collate_fn(sequences, sequences=True, max_seq_len=MAX_SEQ_LEN)
    angles = collate_fn(angles, max_seq_len=MAX_SEQ_LEN)
    coords = collate_fn(coords, coords=True, max_seq_len=MAX_SEQ_LEN)
    return sequences, angles, coords




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
    Includes pad, sos, eos, and unknown characters as well as the 20 standard
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
    def __init__(self, seqs=None, angs=None, crds=None, add_sos_eos=True,
                 sort_by_length=True, reverse_sort=True):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))
        self.vocab = ProteinVocabulary()
        self._seqs = [VOCAB.aa_seq2indices(s, add_sos_eos) for s in seqs]
        self._angs = angs
        self._crds = crds

        if sort_by_length:
            sorted_len_indices = [a[0] for a in sorted(enumerate(angs),
                                                       key=lambda x:x[1].shape[0],
                                                       reverse=reverse_sort)]
            new_seqs = [self._seqs[i] for i in sorted_len_indices]
            self._seqs = new_seqs
            new_angs = [self._angs[i] for i in sorted_len_indices]
            self._angs = new_angs
            new_crds = [self._crds[i] for i in sorted_len_indices]
            self._crds = new_crds


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


class BinnedProteinDataset(torch.utils.data.IterableDataset):
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.
    """
    def __init__(self, seqs=None, angs=None, crds=None, add_sos_eos=True,
                 sort_by_length=True, reverse_sort=True):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))
        self.vocab = ProteinVocabulary()
        self._seqs = [VOCAB.aa_seq2indices(s, add_sos_eos) for s in seqs]
        self._angs = angs
        self._crds = crds

        if sort_by_length:
            sorted_len_indices = [a[0] for a in sorted(enumerate(angs),
                                                       key=lambda x:x[1].shape[0],
                                                       reverse=reverse_sort)]
            new_seqs = [self._seqs[i] for i in sorted_len_indices]
            self._seqs = new_seqs
            new_angs = [self._angs[i] for i in sorted_len_indices]
            self._angs = new_angs
            new_crds = [self._crds[i] for i in sorted_len_indices]
            self._crds = new_crds

        self.lens = sorted(list(map(len, self._seqs)), reverse=reverse_sort)
        self.lens = [l if l < MAX_SEQ_LEN else MAX_SEQ_LEN for l in self.lens]
        self.hist_counts, self.hist_bins = np.histogram(self.lens, bins="auto")
        self.bin_probs = self.hist_counts / self.hist_counts.sum()
        self.bin_map = {}

        for i, s in enumerate(self._seqs):
            for j, bin in enumerate(self.hist_bins):
                if len(s) > bin:
                    continue
                try:
                    self.bin_map[j].append(i)
                except:
                    self.bin_map[j] = [i]

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


class SimilarLengthBatchSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        bin = np.random.choice(range(len(self.data_source.bins)), p=self.data_source.bin_probs)
        return np.random.choice(self.data_source.bin_map[bin], size=self.batch_size)




def prepare_dataloaders(data, args, max_seq_len, num_workers=1):
    """
    Using the pre-processed data, stored in a nested Python dictionary, this
    function returns train, validation, and test set dataloaders with 2 workers
    each. There are multiple validation sets in ProteinNet. Currently, this
    method only returns set '70'.
    """
    sort_data_by_len = args.sort_training_data in ["True", "reverse"]
    reverse_sort = args.sort_training_data == "reverse"

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    # TODO: load and evaluate multiple validation sets
    train_loader = torch.utils.data.DataLoader(
        BinnedProteinDataset(
            seqs=data['train']['seq']*args.repeat_train,
            crds=data['train']['crd']*args.repeat_train,
            angs=data['train']['ang']*args.repeat_train,
            add_sos_eos=args.add_sos_eos,
            sort_by_length=sort_data_by_len,
            reverse_sort=reverse_sort),
        num_workers=num_workers,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=not sort_data_by_len,
        sampler=SimilarLengthBatchSampler)

    valid_loaders = {}
    for split in VALID_SPLITS:
        valid_loader = torch.utils.data.DataLoader(
            ProteinDataset(
                seqs=data[f'valid-{split}']['seq'],
                crds=data[f'valid-{split}']['crd'],
                angs=data[f'valid-{split}']['ang'],
                add_sos_eos=args.add_sos_eos,
                sort_by_length=sort_data_by_len,
                reverse_sort=reverse_sort),
            num_workers=num_workers,
            batch_size=args.batch_size,
            collate_fn=paired_collate_fn,
            worker_init_fn=_init_fn)
        valid_loaders[split] = valid_loader

    test_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['test']['seq'],
            crds=data['test']['crd'],
            angs=data['test']['ang'],
            add_sos_eos=args.add_sos_eos,
            sort_by_length=sort_data_by_len,
            reverse_sort=reverse_sort),
        num_workers=num_workers,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn,
        worker_init_fn=_init_fn)

    return train_loader, valid_loaders, test_loader

VOCAB = ProteinVocabulary()
# TODO remove creation of VOCAB by default