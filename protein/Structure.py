import numpy as np
import torch

import protein.Sidechains as Sidechains
from protein.Sidechains import BONDLENS

BONDLENS = {"n-ca": 1.442, "ca-c": 1.498, "c-n": 1.379,
            "CZ-NH2": 1,
            }


def generate_coords(angles, pad_loc, input_seq, device):
    """ Given a tensor of angles (L x NUM_PREDICTED_ANGLES), produces the entire set of cartesian coordinates using the NeRF method,
        (L x A` x 3), where A` is the number of atoms generated (depends on amino acid sequence)."""
    # TODO: Make the process of grabbing the next residue's nitrogen more clear
    bb_arr = init_backbone(angles, device)
    next_bb_pts = extend_backbone(1, angles, bb_arr, device)
    sc_arr = init_sidechain(angles, [next_bb_pts[0], bb_arr[2], bb_arr[1], bb_arr[0]], input_seq)
    total_arr = bb_arr + sc_arr + (10 - len(sc_arr)) * [torch.zeros(3)]

    for i in range(1, pad_loc):
        bb_pts = extend_backbone(i, angles, bb_arr, device)
        bb_arr += bb_pts

        # Extend sidechain
        sc_pts = Sidechains.extend_sidechain(i, angles, bb_arr, input_seq)

        total_arr += bb_pts + sc_pts + (10 - len(sc_pts)) * [torch.zeros(3)]

    return torch.stack(total_arr)


def generate_coords_with_tuples(angles, pad_loc, input_seq, device):
    """ Identical to generate_cooords, except this function also returns organized tuples of backbone and sidechain
        coordinates to aid in reconstruction.
        Given a tensor of angles (L x NUM_PREDICTED_ANGLES), produces the entire set of cartesian coordinates using the NeRF method,
        (L x A` x 3), where A` is the number of atoms generated (depends on amino acid sequence)."""
    bb_arr = init_backbone(angles, device)
    next_bb_pts = extend_backbone(1, angles, bb_arr, device)
    sc_arr, aa_code, atom_names = init_sidechain(angles, [next_bb_pts[0], bb_arr[2], bb_arr[1], bb_arr[0]], input_seq,
                                                 return_tuples=True)
    aa_codes = [aa_code]
    atom_names_list = [atom_names]
    sc_arr_tups = [[np.asarray(x.detach()) for x in sc_arr[:]]]
    bb_arr_tups = [[np.asarray(x.detach()) for x in bb_arr[:]]]

    for i in range(1, pad_loc):
        bb_pts = extend_backbone(i, angles, bb_arr, device)
        bb_arr += bb_pts

        # Extend sidechain
        sc_pts, aa_code, atom_names = Sidechains.extend_sidechain(i, angles, bb_arr, input_seq, return_tuples=True)
        aa_codes.append(aa_code)
        atom_names_list.append(atom_names)
        sc_arr_tups += [[np.asarray(x.detach()) for x in sc_pts]]
        bb_arr_tups += [[np.asarray(x.detach()) for x in bb_pts]]

        sc_arr += sc_pts

    return torch.stack(bb_arr + sc_arr), [x.detach().numpy() for x in
                                          bb_arr], bb_arr_tups, sc_arr_tups, aa_codes, atom_names_list


def init_sidechain(angles, bb_arr, input_seq, return_tuples=False):
    """ Builds the first sidechain based off of the first backbone atoms and a fictitious previous atom.
        This allows us to reuse the extend_sidechain method. Assumes the first atom of the backbone is at (0,0,0). """
    if return_tuples:
        sc, aa_code, atom_names = Sidechains.extend_sidechain(0, angles, bb_arr, input_seq, return_tuples,
                                                              first_sc=True)
        return sc, aa_code, atom_names
    else:
        sc = Sidechains.extend_sidechain(0, angles, bb_arr, input_seq)
        return sc


def init_backbone(angles, device):
    """ Given an angle matrix (RES x ANG), this initializes the first 3 backbone points (which are arbitrary) and
        returns a TensorArray of the size required to hold all the coordinates. """
    a1 = torch.FloatTensor([0.00001, 0, 0])

    if device.type == "cuda":
        a2 = a1 + torch.cuda.FloatTensor([BONDLENS["n-ca"], 0, 0])
        a3x = torch.cos(np.pi - angles[0, 3]) * BONDLENS["ca-c"]
        a3y = torch.sin(np.pi - angles[0, 3]) * BONDLENS['ca-c']
        a3 = a2 + torch.cuda.FloatTensor([a3x, a3y, 0])
    else:
        a2 = a1 + torch.FloatTensor([BONDLENS["n-ca"], 0, 0])
        a3x = torch.cos(np.pi - angles[0, 3]) * BONDLENS["ca-c"]
        a3y = torch.sin(np.pi - angles[0, 3]) * BONDLENS['ca-c']
        a3 = a2 + torch.FloatTensor([a3x, a3y, 0])

    starting_coords = [a1, a2, a3]

    return starting_coords


def extend_backbone(i, angles, coords, device):
    """ Returns backbone coordinates for the residue angles[pos]."""
    bb_pts = list(coords)
    for j in range(3):
        if j == 0:
            # we are placing N
            t = angles[i - 1, 4]  # thetas["ca-c-n"]
            b = BONDLENS["c-n"]
            dihedral = angles[i - 1, 1]  # psi of previous residue
        elif j == 1:
            # we are placing Ca
            t = angles[i - 1, 5]  # thetas["c-n-ca"]
            b = BONDLENS["n-ca"]
            dihedral = angles[i - 1, 2]  # omega of previous residue
        else:
            # we are placing C
            t = angles[i, 3]  # thetas["n-ca-c"]
            b = BONDLENS["ca-c"]
            dihedral = angles[i, 0]  # phi of current residue
        p3 = bb_pts[-3]
        p2 = bb_pts[-2]
        p1 = bb_pts[-1]
        next_pt = nerf(p3, p2, p1, b, t, dihedral, device)
        bb_pts.append(next_pt)

    return bb_pts[-3:]


def nerf(a, b, c, l, theta, chi, device):
    """ Nerf method of finding 4th coord (d)
        in cartesian space
        Params:
        a, b, c : coords of 3 points
        l : bond length between c and d
        theta : bond angle between b, c, d (in degrees)
        chi : dihedral using a, b, c, d (in degrees)
        Returns:
        d : tuple of (x, y, z) in cartesian space """
    # calculate unit vectors AB and BC
    assert -np.pi <= theta <= np.pi, "theta must be in radians and in [-pi, pi]. theta = " + str(theta)

    W_hat = torch.nn.functional.normalize(b - a, dim=0)
    x_hat = torch.nn.functional.normalize(c-b, dim=0)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = torch.nn.functional.normalize(n_unit, dim=0)
    y_hat = torch.cross(z_hat, x_hat)

    # create rotation matrix [BC; p; n] (3x3)
    M = torch.stack([x_hat, y_hat, z_hat], dim=1)

    # calculate coord pre rotation matrix
    d = torch.stack([torch.squeeze(-l * torch.cos(theta)),
                     torch.squeeze(l * torch.sin(theta) * torch.cos(chi)),
                     torch.squeeze(l * torch.sin(theta) * torch.sin(chi))])

    # calculate with rotation as our final output
    # TODO: is the squeezing necessary?
    d = d.unsqueeze(1)
    res = c + torch.mm(M, d).squeeze()
    return res.squeeze()
