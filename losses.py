import numpy as np
import prody as pr
import torch

from protein.Sidechains import NUM_PREDICTED_ANGLES, NUM_PREDICTED_COORDS
from protein.Structure import generate_coords

BATCH_PAD_CHAR = 0


def combine_drmsd_mse(d, mse, w=.5):
    """ Returns a combination of drmsd and mse loss that first normalizes their scales, and then computes
        w * drmsd + (1 - w) * mse."""
    d_norm, m_norm = 0.01, 0.3
    d = w * (d / d_norm)
    mse = (1 - w) * (mse / m_norm)
    #print(f"mse: {mse:.5f}, d: {d:.5f}")
    return d + mse


def inverse_trig_transform(t):
    """ Given a (BATCH x L X NUM_PREDICTED_ANGLES ) tensor, returns (BATCH X L X NUM_PREDICTED_ANGLES) tensor.
        Performs atan2 transformation from sin and cos values."""
    t = t.view(t.shape[0], -1, NUM_PREDICTED_ANGLES, 2)
    t_cos = t[:, :, :, 0]
    t_sin = t[:, :, :, 1]
    t = torch.atan2(t_sin, t_cos)
    return t


def copy_padding_from_gold(pred, gold, device):
    """ Given two angle tensors, one of which is the true angles (gold) and is properly padded, copy the padding
        from that tensor and apple it to the predicted tensor. The predicted tensor does not understand padding
        sufficiently. """
    # TODO: Assumes any zero is a pad
    not_padded_mask = (gold != 0)
    if device.type == "cuda":
        pred_unpadded = pred.cuda() * not_padded_mask.type(torch.cuda.FloatTensor)
        gold_unpadded = gold.cuda() * not_padded_mask.type(torch.cuda.FloatTensor)
    else:
        pred_unpadded = pred * not_padded_mask.type(torch.FloatTensor)
        gold_unpadded = gold * not_padded_mask.type(torch.FloatTensor)
    return pred_unpadded, gold_unpadded


def drmsd_loss_from_angles(pred, gold, input_seq, device, return_rmsd=False):
    """ Calculate DRMSD loss. """
    raise Exception("Fix pad location before using.")
    device = torch.device("cpu")
    pred, gold = pred.to(device), gold.to(device)

    pred, gold = inverse_trig_transform(pred), inverse_trig_transform(gold)
    pred, gold = copy_padding_from_gold(pred, gold, device)

    losses = []
    rmsds = []
    # TODO: determine which loss functions benefit from GPU vs CPU
    for pred_item, gold_item, input_item in zip(pred, gold, input_seq):
        pad_loc = 0
        gold_item = gold_item[:pad_loc]
        pred_item = pred_item[:pad_loc]
        input_item = input_item[:pad_loc]
        true_coords = generate_coords(gold_item, pad_loc, input_item, device)
        pred_coords = generate_coords(pred_item, pad_loc, input_item, device)
        loss = drmsd(pred_coords, true_coords)
        losses.append(loss)
        if return_rmsd:
            rmsds.append(rmsd(pred_coords.data.numpy(), true_coords.data.numpy()))
    if return_rmsd:
        return torch.mean(torch.stack(losses)), np.mean(rmsds)
    else:
        return torch.mean(torch.stack(losses))


def drmsd_loss_from_coords(pred_angs, gold_coords, input_seqs, device, return_rmsd=False):
    """
    Calculate DRMSD loss by first generating predicted coordinates. Then, these coordinates
    are compared with the true coordinate tensor provided to the function.
    """
    device = torch.device("cpu")
    pred_angs, gold_coords, input_seqs = pred_angs.to(device), gold_coords.to(device), input_seqs.to(device)

    pred_angs = inverse_trig_transform(pred_angs)

    losses = []
    len_normalized_losses = []
    rmsds = []
    for pred_item, gold_item, input_seq in zip(pred_angs, gold_coords, input_seqs):
        batch_mask = input_seq.ne(0).any(dim=1)
        pred_item = pred_item[batch_mask]
        gold_item = gold_item[:pred_item.shape[0]*NUM_PREDICTED_COORDS]
        input_seq = input_seq[batch_mask]
        pred_coords = generate_coords(pred_item, pred_item.shape[0], input_seq, device)
        # TODO the generated coordinates from gold do not match what is expected from the input seq
        gold_item_non_nan = torch.isnan(gold_item).eq(0)
        pred_subset = pred_coords[gold_item_non_nan].reshape(-1, 3)
        gold_subset = gold_item[gold_item_non_nan].reshape(-1, 3)
        loss = drmsd(pred_subset, gold_subset)
        losses.append(loss)
        len_normalized_losses.append(loss / pred_subset.shape[0])
        if return_rmsd:
            rmsds.append(rmsd(pred_subset.data.numpy(), gold_subset.data.numpy()))
    if return_rmsd:
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(len_normalized_losses)), np.mean(rmsds)
    else:
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(len_normalized_losses))


def mse_over_angles(pred, true):
    """ Given a predicted angle tensor and a true angle tensor (batch-padded with zeros,
        and missing-item-padded with nans), this function first removes batch then item
        padding before using torch's built-in MSE loss function. """
    ang_non_zero = true.ne(0).any(dim=2)
    tgt_ang_non_zero = true[ang_non_zero]
    ang_non_nans = torch.isnan(tgt_ang_non_zero).eq(0)
    return torch.nn.functional.mse_loss(pred[ang_non_zero][ang_non_nans], true[ang_non_zero][ang_non_nans])


def my_cdist(x1, x2):
    # TODO cite source from Pytorch forum / github
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def pairwise_internal_dist(coords):
    """ Returns a tensor of the pairwise distances between all points in coords. """
    return my_cdist(coords, coords)


def drmsd(a, b, pad_from_b=False):
    """ Given two coordinate tensors, returns the dRMSD score between them.
        Both tensors must be the exact same shape. """
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
    """ Returns the RMSD between two sets of coordinates."""
    t = pr.calcTransformation(a, b)
    return pr.calcRMSD(t.apply(a), b)
    

def drmsd_loss_rnn(y_pred_, y_true_, sorted_lengths, x):
    """ Given angle Tensors, return the drmsd loss. (Batch x L x NUM_PREDICTED_ANGLES x 2)"""

    y_pred_ = torch.atan2(y_pred_[:, :, :, 1], y_pred_[:, :, :, 0])
    y_true_ = torch.atan2(y_true_[:, :, :, 1], y_true_[:, :, :, 0])

    def work(pred, true, length, input_seq):
        """ Takes single protein predicted and true angle tensors, along with their actual length and amino acid
            sequence. Truncates tensor to correct length, generates cartesian coords, then computes drmsd."""
        pred = pred[:length]
        true = true[:length]
        pred = generate_coords(pred, input_seq)
        true = generate_coords(true, input_seq)
        return drmsd(pred, true)

    def curry_work(p_t_l_i):
        """ Curries the work function for use in python map statement. """
        return work(p_t_l_i[0], p_t_l_i[1], p_t_l_i[2], p_t_l_i[3])

    drmsds = list(map(curry_work, zip(y_pred_, y_true_, sorted_lengths, x)))
    drmsds = torch.stack(drmsds)

    # Non-working parallel implementation
    # p = Pool(mp.cpu_count())
    # drmsds = torch.stack(p.map(curry_work, zip(y_pred_, y_true_, sorted_lengths)))

    return drmsds.mean()

