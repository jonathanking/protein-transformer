from protein_transformer.losses import *
import numpy as np


def test_mse_loss():
    random = np.random.random((8, 15, 24))
    a = torch.tensor(random)
    b = torch.tensor(random)
    assert mse_over_angles(a, b) == 0



