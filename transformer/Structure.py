import torch
import numpy as np
import torch.nn.functional as F


def angles2coords(angles):
    """ Given an angle tensor, returns a coordinate tensor."""
    coords = initialize_backbone_array(angles)

    for i in range(1, len(angles)):
        if angles[i].sum() == 0:
            break
        coords = extend_backbone(i, angles, coords)
    return torch.stack(coords)


def initialize_backbone_array(angles):
    """ Given an angle matrix (RES x ANG), this initializes the first 3 backbone points (which are arbitrary) and
        returns a TensorArray of the size required to hold all the coordinates. """
    bondlens = {"n-ca": 1.442, "ca-c": 1.498, "c-n": 1.379}

    a1 = torch.zeros(3)
    a2 = a1 + torch.FloatTensor([bondlens["n-ca"], 0, 0])
    a3x = torch.cos(np.pi - angles[0,3]) * bondlens["ca-c"]
    a3y = torch.sin(np.pi - angles[0,3]) * bondlens['ca-c']
    a3 = torch.FloatTensor([a3x, a3y, 0])
    starting_coords = [a1, a2, a3]

    return starting_coords


def extend_backbone(i, angles, coords):
    """ Returns backbone coordinates for the residue angles[pos]."""
    bondlens = {"n-ca": 1.442, "ca-c": 1.498, "c-n": 1.379}

    for j in range(3):
        if j == 0:
            # we are placing N
            t = angles[i, 4]  # thetas["ca-c-n"]
            b = bondlens["c-n"]
            dihedral = angles[i - 1, 1]  # psi of previous residue
        elif j == 1:
            # we are placing Ca
            t = angles[i, 5]  # thetas["c-n-ca"]
            b = bondlens["n-ca"]
            dihedral = angles[i - 1, 2]  # omega of previous residue
        else:
            # we are placing C
            t = angles[i, 3]  # thetas["n-ca-c"]
            b = bondlens["ca-c"]
            dihedral = angles[i, 0]  # phi of current residue
        p3 = coords[-3]
        p2 = coords[-2]
        p1 = coords[-1]
        next_pt = nerf(p3, p2, p1, b, t, dihedral)
        coords.append(next_pt)


    return coords


def l2_normalize(t, eps=1e-12):
    epsilon = torch.FloatTensor([eps])
    return t / torch.sqrt(torch.max((t**2).sum(), epsilon))


def nerf(a, b, c, l, theta, chi):
    '''
    Nerf method of finding 4th coord (d)
    in cartesian space
    Params:
    a, b, c : coords of 3 points
    l : bond length between c and d
    theta : bond angle between b, c, d (in degrees)
    chi : dihedral using a, b, c, d (in degrees)
    Returns:
    d : tuple of (x, y, z) in cartesian space
    '''
    # calculate unit vectors AB and BC

    W_hat = l2_normalize(b - a)
    x_hat = l2_normalize(c - b)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = l2_normalize(n_unit)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=1)

    # calculate coord pre rotation matrix
    d = torch.stack([torch.squeeze(-l * torch.cos(theta)),
                     torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
                     torch.squeeze(l * torch.sin(theta) * torch.sin(chi))])

    # calculate with rotation as our final output

    d = d.unsqueeze(1)

    res = c + torch.mm(M, d).squeeze()

    return res.squeeze()


def drmsd(a, b):
    """ Given two coordinate tensors, returns the dRMSD score between them.
        Both tensors must be the exact same shape. """

    a_ = pairwise_internal_dist(a)
    b_ = pairwise_internal_dist(b)
    return torch.sqrt(F.mse_loss(a_, b_))

    # num_elems = a_.shape[0]
    #
    # # Option 1 - RMSD as expected. Sqrt placement can be altered to mimic MAQ's description below.
    # sq_diff = (a_**2) - (b_**2)
    # summed = sq_diff.sum()
    # mean = summed / num_elems
    # res = torch.sqrt(mean)
    #
    #
    # return res


def pairwise_internal_dist(coords):
    """ Returns a tensor of the pairwise distances between all points in coords. """
    c1 = coords.unsqueeze(1)
    c2 = coords.unsqueeze(0)
    z = c1 - c2 + 1e-10  # (L x L x 3)
    res = torch.norm(z, dim=2)         # (L x L)
    return res