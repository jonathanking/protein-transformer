import numpy as np
import pytest
import torch

from protein_transformer.losses import *


@pytest.mark.parametrize("drmsd, mse, w, expected",[
    (0.01, 0.3, 0.5, 1),
    (0.01, 0.6, 0, 2),
    (0.02, 0.3, 1, 2)  
])
def test_combine_drmsd_mse(drmsd, mse, w, expected):
    assert combine_drmsd_mse(drmsd, mse, w) == expected

def test_inverse_trig_transform():
    pass

def test_mse_over_angles():
    random = np.random.random((8, 15, 24))
    a = torch.tensor(random)
    b = torch.tensor(random)
    assert mse_over_angles(a, b) == 0


######### DRMSD #########

def test_drmsd_zero():
    a = np.asarray([[0,0,0], [3, 5, 2], [2, 9, 3]])
    b = np.asarray([[0,0,0], [3, 5, 2], [2, 9, 3]])
    a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    assert drmsd(a, b) == 0

### DRMSD Helper Tests ###

def compute_intra_array_distances(x):
    deltas = []
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[0]):
            deltas.append(np.linalg.norm(x[i] - x[j]))
    return np.asarray(deltas)

def test_compute_intra_array_distances():
    a = np.asarray([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])
    distances = np.asarray([1, 1, np.sqrt(2), np.sqrt(2), 1, 1])
    assert pytest.approx(compute_intra_array_distances(a)) == distances


def lazy_drmsd(a, b):
    """ 
    A lazy version of DRMSD. First computes intra-array distances. Then
    computes RMSE between those arrays. 
    """

    a_deltas, b_deltas = compute_intra_array_distances(a), compute_intra_array_distances(b)

    total_delta_delta = 0.0
    n = 0.0
    for ad, bd in zip(a_deltas, b_deltas):
        sqdiff = (ad-bd)**2
        total_delta_delta += sqdiff
        n += 1

    total_delta_delta /= n
    return np.sqrt(total_delta_delta)


def test_drmsd_equals_lazy_drmsd():
    
    x = np.asarray([[0,0,0], [3, 5, 2], [2, 9, 3]])
    y = np.asarray([[0,0,0], [9, 3, 1], [4, 7, 8]])    
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    assert pytest.approx(drmsd(x, y).item()) == lazy_drmsd(x, y)

def test_pairwise_internal_dist():
    pass

def test_mse_over_angles():
    pass







