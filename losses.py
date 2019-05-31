import numpy as np
import prody as pr
import torch
import torch.nn.functional as F

from transformer.Sidechains import NUM_PREDICTED_ANGLES
from transformer.Structure import generate_coords


def combine_drmsd_mse(d, mse, w=.5):
    """ Returns a combination of drmsd and mse loss that first normalizes their scales, and then computes
        w * drmsd + (1 - w) * mse."""
    d_norm, m_norm = 8, 2
    d = d / d_norm
    mse = mse / m_norm
    return w * d + (1 - w) * mse


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
    not_padded_mask = (gold != 0)
    if device.type == "cuda":
        pred_unpadded = pred.cuda() * not_padded_mask.type(torch.cuda.FloatTensor)
        gold_unpadded = gold.cuda() * not_padded_mask.type(torch.cuda.FloatTensor)
    else:
        pred_unpadded = pred * not_padded_mask.type(torch.FloatTensor)
        gold_unpadded = gold * not_padded_mask.type(torch.FloatTensor)
    return pred_unpadded, gold_unpadded


def determine_pad_loc(padded_tensor):
    """ Returns the position where the tensor becomes padded with all 0s across the first dimension. """
    paddings = torch.all(padded_tensor == 0, dim=1)
    if torch.all(paddings == 0).item() == 1:
        return padded_tensor.shape[0]
    else:
        return torch.argmin(paddings) + 1


def determine_pad_loc_test():
    """ Test for determin_pad_loc. """
    # TODO use more official test suite
    a = np.random.rand(54, 12)
    b = np.random.rand(63, 12)
    a[11:] = np.zeros(shape=(a.shape[0] - 11, 12))
    b[33:] = np.zeros(shape=(b.shape[0] - 33, 12))
    at = torch.tensor(a)
    bt = torch.tensor(b)
    assert determine_pad_loc(at).item() == 11
    assert determine_pad_loc(bt).item() == 33


def drmsd_loss(pred, gold, input_seq, device, return_rmsd=False):
    """ Calculate DRMSD loss. """
    device = torch.device("cpu")
    pred, gold = pred.to(device), gold.to(device)

    pred, gold = inverse_trig_transform(pred), inverse_trig_transform(gold)
    pred, gold = copy_padding_from_gold(pred, gold, device)

    losses = []
    rmsds = []
    # TODO: gracefully handle losses when batchsize is 1.
    if pred.shape[0] == 1:
        pred_item = pred[0]
        gold_item = gold[0]
        input_item = input_seq[0]
        true_coords = generate_coords(gold_item, pred_item.shape[0], input_item, device)
        pred_coords = generate_coords(pred_item, pred_item.shape[0], input_item, device)
        loss = drmsd(pred_coords, true_coords)
        losses.append(loss)
        if return_rmsd:
            rmsds.append(rmsd(pred_coords.data.numpy(), true_coords.data.numpy()))
    else:
        for pred_item, gold_item, input_item in zip(pred, gold, input_seq):
            pad_loc = determine_pad_loc(gold_item)
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


def mse_loss(pred, gold):
    """ Computes MSE loss."""
    device = torch.device("cpu")
    pred, gold = pred.to(device), gold.to(device)
    pred, gold = inverse_trig_transform(pred), inverse_trig_transform(gold)
    pred, gold = copy_padding_from_gold(pred, gold, device)
    mse = F.mse_loss(pred, gold)

    return mse


def pairwise_internal_dist(coords):
    """ Returns a tensor of the pairwise distances between all points in coords. """
    c1 = coords.unsqueeze(1)
    c2 = coords.unsqueeze(0)
    z = c1 - c2 + 1e-10        # (L x L x 3)
    res = torch.norm(z, dim=2)  # (L x L)
    return res


def drmsd(a, b):
    """ Given two coordinate tensors, returns the dRMSD score between them.
        Both tensors must be the exact same shape. """
    mask = a.ne(0).any(1)
    a = a[mask]
    mask = b.ne(0).any(1)
    b = b[mask]

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


if __name__ == "__main__":
    determine_pad_loc_test()
