import numpy as np
import torch
from pytest import approx
import pytest

from protein_transformer.dataset import BinnedProteinDataset, paired_collate_fn, SimilarLengthBatchSampler

from protein_transformer.protein.Structure import NUM_PREDICTED_ANGLES, \
    NUM_PREDICTED_COORDS

np. set_printoptions(precision=None, suppress=True)

# @pytest.fixture
# def casp12_dataset_ex():
#     d = torch.load("/home/jok120/protein-transformer/data/proteinnet/casp12_200123_30.pt")
#     return d



def test_BinnedProteinDataset():
    seqs = ["SDFHUDHHHB", "SDKFJHSDKFSKJD", "SDKFJHSDKFSKJD",
            "SDFJOWIEJROWJOJLLOJOWIERJWLKJFSMDSF"]
    angs = [np.random.rand(len(s), NUM_PREDICTED_ANGLES, 2) for s in seqs]
    crds = [np.random.rand(len(s), NUM_PREDICTED_COORDS, 3) for s in seqs]

    bpd = BinnedProteinDataset(seqs, angs, crds, add_sos_eos=False)

    assert approx(bpd.bin_probs.sum()) == 1  # all bin probs sum to one
    assert any(1 in v and 2 in v for v in bpd.bin_map.values())  # one bin has two seqs (identical length)
    assert 0 in bpd.bin_map[0]  # first seq in first bin
    assert len(seqs) - 1 in bpd.bin_map[max(bpd.bin_map.keys())] # last seq in last bin


# def test_BinnedProteinDataset_200122dataset(casp12_dataset_ex):
#     d = casp12_dataset_ex
#     seqs, angs, crds = d["train"]["seq"], d["train"]["ang"], d["train"]["crd"]
#
#
#     bpd = BinnedProteinDataset(seqs, angs, crds, add_sos_eos=False)
#
#     assert approx(bpd.bin_probs.sum()) == 1  # all bin probs sum to one
#     assert 0 in bpd.bin_map[0]  # first seq in first bin
#     assert len(seqs) - 1 in bpd.bin_map[max(bpd.bin_map.keys())] # last seq in last bin
#
#
# def test_BinnedProteinDataset_sampling(casp12_dataset_ex):
#     d = casp12_dataset_ex
#     seqs, angs, crds = d["train"]["seq"], d["train"]["ang"], d["train"]["crd"]
#
#     train_dataset = BinnedProteinDataset(
#         seqs=seqs,
#         crds=crds,
#         angs=angs,
#         add_sos_eos=False)
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         num_workers=1,
#         collate_fn=paired_collate_fn,
#         batch_sampler=SimilarLengthBatchSampler(train_dataset, 12))
#
#     batcher = iter(train_loader)
#     b = next(batcher)
#     bseqs, bangs, bcrds = b
#     # TODO finish writing test for sampling

