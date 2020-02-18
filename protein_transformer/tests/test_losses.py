import numpy as np
import pytest
import torch
from pytest import approx

from protein_transformer.losses import *

np. set_printoptions(precision=None, suppress=True)


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
    """
    Given an array of coordinates, compute, in order, the differences between
    those coordinates.
    """
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

@pytest.mark.parametrize("x, y",[
    (
        np.asarray([[0,0,0], [3, 5, 2], [2, 9, 3]]),
        np.asarray([[0,0,0], [9, 3, 1], [4, 7, 8]])
        ),
    (
        np.random.random((50,3))*10,
        np.random.random((50,3))*10
        )
])
def test_drmsd_equals_lazy_drmsd(x, y):
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    assert pytest.approx(drmsd(x, y).item()) == lazy_drmsd(x, y)

def test_pairwise_internal_dist():
    a = np.asarray([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])
    a = torch.tensor(a, dtype=torch.float32)
    distances = np.asarray([1, 1, np.sqrt(2), np.sqrt(2), 1, 1])

    pwd_result = pairwise_internal_dist(a)[0].numpy()
    print(pwd_result)
    print(pwd_result.shape)
    assert pytest.approx(pairwise_internal_dist(a)[0].numpy()) == distances

def test_pairwise_internal_dist2():
    a = np.asarray([[0], [1], [2]])
    a = torch.tensor(a, dtype=torch.float32)
    distances = np.asarray([1, 2, 1])

    pwd_result = pairwise_internal_dist(a)[0].numpy()
    print(pwd_result)
    print(pwd_result.shape)
    assert pytest.approx(pwd_result.sum()) == distances.sum()


def test_mse_over_angles():
    pass






def test_mse_loss2():
    a = np.zeros((8,10,24))
    b = a - .1
    assert mse_over_angles_numpy(a, b) == approx(0.01)



def test_pairwise_internal_dist():
    a = np.asarray([[0,0,0],
                    [0,1,0],
                    [0,0,2],
                    [0,0,0]])
    a = torch.tensor(a, dtype=torch.float)

    correct = [[0,1,     2,    0],
               [1,0,     2.236,1],
               [2,2.236, 0,    2],
               [0,1,     2,    0]]
    correct = np.asarray(correct)

    pid = pairwise_internal_dist(a).numpy()
    assert pid == approx(correct, rel=1e-3)


def test_pairwise_internal_dist2():
    a = np.asarray([[5.3,-15.2,300],
                    [-3.3, 234.1, 0]])
    a = torch.tensor(a, dtype=torch.float)

    correct = [[0, 390.15951865],
               [390.15951865, 0]]
    correct = np.asarray(correct)

    pid = pairwise_internal_dist(a).numpy()
    assert pid == approx(correct)


def test_drmsd_zero():
    a = np.asarray([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 2],
                    [0, 0, 0]])
    a = torch.tensor(a, dtype=torch.float)
    assert drmsd(a, a) == 0
    assert drmsd(a, a, truncate_dist_matrix=False) == 0


def test_drmsd_permutation():
    """ DRMSD is not permutation invariant. """
    a = np.asarray([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 2],
                    [0, 0, 0]])
    b = np.asarray([[0, 0, 2],
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    a = torch.tensor(a, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    assert drmsd(a, b) != 0
    assert drmsd(a, b, truncate_dist_matrix=False) != 0


@pytest.mark.parametrize(
    "a,b",
     [
         (np.array([[0.  , 2.8 , 2.95],
                    [2.45, 3.35, 4.4 ],
                    [4.3 , 2.  , 0.55],
                    [0.9 , 3.75, 2.05],
                    [0.35, 3.  , 1.25]]),
          np.array([[0.75, 4.5, 1.5],
                    [2.85, 4.85, 0.9],
                    [4.65, 1.7, 0.65],
                    [1.55, 1.5, 1.15],
                    [4.4, 1.15, 3.2]])
          ),
         (np.array([[ 6.1,  0.2,  6.4],
                    [ 2.2,  4.6, -1.7],
                    [ 4.5,  2.6,  3.1],
                    [ 1.1, -0.3,  2.3],
                    [-0.2,  7.3,  3.6]]),
          np.array([[-1.1,  2.5,  6.1],
                    [ 7.4, -1.6,  6.4],
                    [ 1.3,  1.2,  1.7],
                    [-0.7,  1.7,  6. ],
                    [-0.4,  4.2,  2.9]])
          )
     ]
)
def test_drmsd_value_symmetric_distances(a, b):
    """
    Tests that DRMSD works when the symmetric distance matrix is not truncated.
    """
    a_dists = []
    for c1 in a:
        for c2 in a:
            a_dists.append(np.linalg.norm(c1-c2))
    b_dists = []
    for c1 in b:
        for c2 in b:
            b_dists.append(np.linalg.norm(c1 - c2))
    a_dists, b_dists = np.asarray(a_dists), np.asarray(b_dists)

    mse = ((a_dists - b_dists) ** 2).mean()
    expected_drmsd = np.sqrt(mse)

    assert drmsd(torch.tensor(a), torch.tensor(b), truncate_dist_matrix=False).item() == approx(expected_drmsd)


@pytest.mark.parametrize(
    "a,b",
     [
         (np.array([[0.  , 2.8 , 2.95],
                    [2.45, 3.35, 4.4 ],
                    [4.3 , 2.  , 0.55],
                    [0.9 , 3.75, 2.05],
                    [0.35, 3.  , 1.25]]),
          np.array([[0.75, 4.5, 1.5],
                    [2.85, 4.85, 0.9],
                    [4.65, 1.7, 0.65],
                    [1.55, 1.5, 1.15],
                    [4.4, 1.15, 3.2]])
          ),
         (np.array([[ 6.1,  0.2,  6.4],
                    [ 2.2,  4.6, -1.7],
                    [ 4.5,  2.6,  3.1],
                    [ 1.1, -0.3,  2.3],
                    [-0.2,  7.3,  3.6]]),
          np.array([[-1.1,  2.5,  6.1],
                    [ 7.4, -1.6,  6.4],
                    [ 1.3,  1.2,  1.7],
                    [-0.7,  1.7,  6. ],
                    [-0.4,  4.2,  2.9]])
          )
     ]
)


def test_drmsd_value_nonsymmetric_distances(a, b):
    """
    Tests that DRMSD works when the symmetric distance matrix is truncated.
    """
    a_dists = []
    for i, c1 in enumerate(a):
        for j, c2 in enumerate(a):
            if j > i:
                a_dists.append(np.linalg.norm(c1-c2))

    assert len(a_dists) == a.shape[0] * (a.shape[0] - 1) / 2


    b_dists = []
    for i, c1 in enumerate(b):
        for j, c2 in enumerate(b):
            if j > i:
                b_dists.append(np.linalg.norm(c1 - c2))

    a_dists, b_dists = np.asarray(a_dists), np.asarray(b_dists)
    mse = ((a_dists - b_dists)**2).mean()
    expected_drmsd = np.sqrt(mse)

    assert drmsd(torch.tensor(a), torch.tensor(b),
                 truncate_dist_matrix=True).item() == approx(expected_drmsd)