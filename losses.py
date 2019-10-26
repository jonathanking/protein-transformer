""" Loss functions for training protein structure prediction models. """

import numpy as np
import prody as pr
import torch

from protein.Sidechains import NUM_PREDICTED_ANGLES, NUM_PREDICTED_COORDS
from protein.Structure import generate_coords

BATCH_PAD_CHAR = 0


def combine_drmsd_mse(d, mse, w=.5):
    """
    Returns a combination of drmsd and mse loss that first normalizes their
    zscales, and then computes w * drmsd + (1 - w) * mse.
    """
    d_norm, m_norm = 0.01, 0.3
    d = w * (d / d_norm)
    mse = (1 - w) * (mse / m_norm)
    return d + mse


def inverse_trig_transform(t):
    """
    Given a (BATCH x L X NUM_PREDICTED_ANGLES ) tensor, returns (BATCH X
    L X NUM_PREDICTED_ANGLES) tensor. Performs atan2 transformation from sin
    and cos values.
    """
    t = t.view(t.shape[0], -1, NUM_PREDICTED_ANGLES, 2)
    t_cos = t[:, :, :, 0]
    t_sin = t[:, :, :, 1]
    t = torch.atan2(t_sin, t_cos)
    return t


def copy_padding_from_gold(pred, gold, device):
    """
    Given two angle tensors, one of which is the true angles (gold) and
    is properly padded, copy the padding from that tensor and apple it to the
    predicted tensor. The predicted tensor does not understand padding
    sufficiently.
    """
    # TODO: Assumes any zero is a pad
    not_padded_mask = (gold != 0)
    if device.type == "cuda":
        pred_unpadded = pred.cuda() * not_padded_mask.type(torch.cuda.FloatTensor)
        gold_unpadded = gold.cuda() * not_padded_mask.type(torch.cuda.FloatTensor)
    else:
        pred_unpadded = pred * not_padded_mask.type(torch.FloatTensor)
        gold_unpadded = gold * not_padded_mask.type(torch.FloatTensor)
    return pred_unpadded, gold_unpadded


def drmsd_loss_from_angles(pred_angs, true_crds, input_seqs, device, return_rmsd=False):
    """
    Calculate DRMSD loss by first generating predicted coordinates from
    angles. Then, predicted coordinates are compared with the true coordinate
    tensor provided to the function.
    """
    device = torch.device("cpu")
    pred_angs, true_crds, input_seqs = pred_angs.to(device), true_crds.to(device), input_seqs.to(device)

    pred_angs = inverse_trig_transform(pred_angs)

    losses = []
    len_normalized_losses = []
    rmsds = []

    for pred_ang, true_crd, input_seq in zip(pred_angs, true_crds, input_seqs):
        # Remove batch-level masking
        batch_mask = input_seq.ne(0).any(dim=1)
        pred_ang = pred_ang[batch_mask]
        true_crd = true_crd[:pred_ang.shape[0] * NUM_PREDICTED_COORDS]
        input_seq = input_seq[batch_mask]

        # Generate coordinates
        pred_crd = generate_coords(pred_ang, pred_ang.shape[0], input_seq, device)

        # Remove coordinate-level masking for missing atoms
        true_crd_non_nan = torch.isnan(true_crd).eq(0)
        pred_crds_masked = pred_crd[true_crd_non_nan].reshape(-1, 3)
        true_crds_masked = true_crd[true_crd_non_nan].reshape(-1, 3)

        # Compute drmsd between existing atoms only
        loss = drmsd(pred_crds_masked, true_crds_masked)

        # Record-keeping; return rmsd if requested
        losses.append(loss)
        len_normalized_losses.append(loss / pred_crds_masked.shape[0])
        if return_rmsd:
            rmsds.append(rmsd(pred_crds_masked.data.numpy(), true_crds_masked.data.numpy()))

    if return_rmsd:
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(len_normalized_losses)), np.mean(rmsds)
    else:
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(len_normalized_losses))


def mse_over_angles(pred, true):
    """
    Given a predicted angle tensor and a true angle tensor (batch-padded with
    zeros, and missing-item-padded with nans), this function first removes
    batch then item padding before using torch's built-in MSE loss function.
    """
    ang_non_zero = true.ne(0).any(dim=2)
    tgt_ang_non_zero = true[ang_non_zero]
    ang_non_nans = torch.isnan(tgt_ang_non_zero).eq(0)
    return torch.nn.functional.mse_loss(pred[ang_non_zero][ang_non_nans], true[ang_non_zero][ang_non_nans])


def pairwise_internal_dist(x):
    """
    An implementation of cdist (pairwise distances between sets of vectors)
    from user jacobrgardner on github. Not implemented for batches.

    https://github.com/pytorch/pytorch/issues/15253
    """
    x1, x2 = x, x
    assert len(x1.shape) == 2, "Pairwise internal distance method is not " \
                               "implemented for batches."
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def drmsd(a, b):
    """
    Given two coordinate tensors, returns the dRMSD score between them. Both
    tensors must be the exact same shape.
    """
    a_ = pairwise_internal_dist(a)
    b_ = pairwise_internal_dist(b)

    num_elems = a_.shape[0]
    num_elems = num_elems * (num_elems - 1)

    sq_diff = (a_ - b_) ** 2
    summed = sq_diff.sum()
    mean = summed / num_elems
    res = mean.sqrt()

    return res


def rmsd(a, b):
    """
    Returns the RMSD between two sets of coordinates.
    """
    t = pr.calcTransformation(a, b)
    return pr.calcRMSD(t.apply(a), b)
