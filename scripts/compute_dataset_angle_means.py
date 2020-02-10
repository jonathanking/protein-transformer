import torch
import numpy as np

import sys

from protein_transformer.losses import inverse_trig_transform


def main():
    """
    Computes the average values for the angles of the training set. Requires
    as input two strings: in dataset path, and the outpath to save the numpy
    array of the mean of the training angles. This data gets loaded by the model
    during weight initialization.
    """
    data = torch.load(sys.argv[1])
    train_angles_sincos = np.concatenate(data["train"]["ang"])
    train_angles = numpy_inverse_trig(train_angles_sincos)
    new_means = np.nanmean(train_angles, axis=0)
    np.save(sys.argv[2], new_means)

def numpy_inverse_trig(arr):
    return inverse_trig_transform(torch.tensor(np.expand_dims(arr, axis=0))).numpy()[0]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please provide two arguments: path to torch dataset, outpath for angle means.")
        sys.exit(1)
    main()