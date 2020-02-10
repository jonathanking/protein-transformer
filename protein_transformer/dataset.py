import numpy as np
import torch
import torch.utils.data
import wandb

from protein_transformer.protein.Sequence import ProteinVocabulary, VOCAB
from protein_transformer.protein.Structure import NUM_PREDICTED_COORDS

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


class ProteinDataset(torch.utils.data.Dataset):
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.
    """
    def __init__(self, seqs=None, angs=None, crds=None, add_sos_eos=True,
                 sort_by_length=True, reverse_sort=True, skip_missing_residues=True):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))
        self._seqs, self._angs, self._crds = [], [], []
        for i in range(len(seqs)):
            if np.isnan(angs[i]).all(axis=-1).any() and skip_missing_residues:
                continue
            else:
                self._seqs.append(VOCAB.str2ints(seqs[i], add_sos_eos))
                self._angs.append(angs[i])
                self._crds.append(crds[i])


        if sort_by_length:
            sorted_len_indices = [a[0] for a in sorted(enumerate(self._angs),
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


class BinnedProteinDataset(torch.utils.data.Dataset):
    """
    This dataset can hold lists of sequences, angles, and coordinates for
    each protein.

    Assumes protein data is sorted from shortest to longest (ascending).
    """
    def __init__(self, seqs=None, angs=None, crds=None, add_sos_eos=True, skip_missing_residues=True):

        assert seqs is not None
        assert (angs is None) or (len(seqs) == len(angs) and len(angs) == len(crds))
        self.vocab = ProteinVocabulary()
        self._seqs, self._angs, self._crds = [], [], []
        for i in range(len(seqs)):
            if np.isnan(angs[i]).all(axis=-1).any() and skip_missing_residues:
                continue
            else:
                self._seqs.append(VOCAB.str2ints(seqs[i], add_sos_eos))
                self._angs.append(angs[i])
                self._crds.append(crds[i])


        # Compute length-based histogram bins and probabilities
        self.lens = list(map(lambda x: len(x) if len(x) <= MAX_SEQ_LEN else MAX_SEQ_LEN, self._seqs))
        self.hist_counts, self.hist_bins = np.histogram(self.lens, bins="auto")
        self.hist_bins = self.hist_bins[1:]  # make each bin define the rightmost value in each bin, ie '( , ]'.
        self.bin_probs = self.hist_counts / self.hist_counts.sum()
        self.bin_map = {}

        # Compute a mapping from bin number to index in dataset
        seq_i = 0
        bin_j = 0
        while seq_i < len(self._seqs):
            if self.lens[seq_i] <= self.hist_bins[bin_j]:
                try:
                    self.bin_map[bin_j].append(seq_i)
                except KeyError:
                    self.bin_map[bin_j] = [seq_i]
                seq_i += 1
            else:
                bin_j += 1


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
    """
    When turned into an iterator, this Sampler is designed to yield batches
    of indices at a time. The indices correspond to sequences in the dataset,
    grouped by sequence length. This has the effect of yielding batches where
    all items in a batch have similar length, but average length of any
    particular batch is completely random.
    """

    def __init__(self, data_source, batch_size, dynamic_batch):
        self.data_source = data_source
        self.batch_size = batch_size
        self.dynamic_batch = dynamic_batch

    def __len__(self):
        # If batches are dynamically sized to contain the same number of residues,
        # then the approximate number of batches is the total number of residues in the dataset
        # divided by the size of the dynamic batch.
        if self.dynamic_batch:
            return int(np.ceil(sum(self.data_source.lens) / self.dynamic_batch))
        return int(np.ceil(len(self.data_source) / self.batch_size))

    def __iter__(self):
        def batch_generator():
            i = 0
            while i < len(self):
                bin = np.random.choice(range(len(self.data_source.hist_bins)), p=self.data_source.bin_probs)
                if self.dynamic_batch:
                    this_batch_size = int(self.dynamic_batch / self.data_source.hist_bins[bin])
                else:
                    this_batch_size = self.batch_size
                wandb.log({"batch_size": this_batch_size}, commit=False)
                yield np.random.choice(self.data_source.bin_map[bin], size=this_batch_size)
                i += 1
        return batch_generator()



def prepare_dataloaders(data, args, max_seq_len, num_workers=1):
    """
    Using the pre-processed data, stored in a nested Python dictionary, this
    function returns train, validation, and test set dataloaders with 2 workers
    each. Note that there are multiple validation sets in ProteinNet.
    """

    if args.batching_order in ["descending", "ascending"]:
        raise NotImplementedError("Descending and ascending order have not been reimplemented.")

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    train_dataset = BinnedProteinDataset(
            seqs=data['train']['seq']*args.repeat_train,
            crds=data['train']['crd']*args.repeat_train,
            angs=data['train']['ang']*args.repeat_train,
            add_sos_eos=args.add_sos_eos, skip_missing_residues=args.skip_missing_res_train)
    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    num_workers=num_workers,
                    collate_fn=paired_collate_fn,
                    batch_sampler=SimilarLengthBatchSampler(train_dataset,
                                                            args.batch_size,
                                                            dynamic_batch=args.batch_size * MAX_SEQ_LEN))
    train_eval_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        collate_fn=paired_collate_fn,
                        batch_size=args.batch_size)

    valid_loaders = {}
    for split in VALID_SPLITS:
        valid_loader = torch.utils.data.DataLoader(
            ProteinDataset(
                seqs=data[f'valid-{split}']['seq'],
                crds=data[f'valid-{split}']['crd'],
                angs=data[f'valid-{split}']['ang'],
                add_sos_eos=args.add_sos_eos,
                skip_missing_residues=args.skip_missing_res_train),
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
            skip_missing_residues=args.skip_missing_res_train),
        num_workers=num_workers,
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn,
        worker_init_fn=_init_fn)

    return train_loader, train_eval_loader, valid_loaders, test_loader
