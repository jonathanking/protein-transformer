import numpy as np
import torch

import transformer.Sidechains as Sidechains

BONDLENS = {"n-ca": 1.442, "ca-c": 1.498, "c-n": 1.379,
            "CZ-NH2": 1,
            }


def generate_coords(angles, pad_loc, input_seq, device):
    """ Given a tensor of angles (L x 11), produces the entire set of cartesian coordinates using the NeRF method,
        (L x A` x 3), where A` is the number of atoms generated (depends on amino acid sequence)."""
    bb_arr = init_backbone(angles, device)
    sc_arr = init_sidechain(angles, bb_arr, input_seq)

    for i in range(1, pad_loc):
        bb_pts = extend_backbone(i, angles, bb_arr, device)
        bb_arr += bb_pts

        # Extend sidechain
        sc_pts = Sidechains.extend_sidechain(i, angles, bb_arr, input_seq)
        sc_arr += sc_pts

    return torch.stack(bb_arr + sc_arr)


def generate_coords_with_tuples(angles, pad_loc, input_seq, device):
    """ Identical to generate_cooords, except this function also returns organized tuples of backbone and sidechain
        coordinates to aid in reconstruction.
        Given a tensor of angles (L x 11), produces the entire set of cartesian coordinates using the NeRF method,
        (L x A` x 3), where A` is the number of atoms generated (depends on amino acid sequence)."""
    bb_arr = init_backbone(angles, device)
    sc_arr, aa_code, atom_names = init_sidechain(angles, bb_arr, input_seq, return_tuples=True)
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
    """ Builds the first sidechain based off of the first backbone atoms and a ficticious previous atom.
        This allows us to reuse the extend_sidechain method. Assumes the first atom of the backbone is at (0,0,0). """
    fake_prev_c = torch.FloatTensor(torch.zeros(3))
    fake_prev_c[0] = -np.cos(np.pi - 122) * BONDLENS["c-n"]
    fake_prev_c[1] = -np.sin(np.pi - 122) * BONDLENS["c-n"]
    if return_tuples:
        sc, aa_code, atom_names = Sidechains.extend_sidechain(0, angles, [fake_prev_c] + bb_arr, input_seq,
                                                              return_tuples)
        return sc, aa_code, atom_names
    else:
        sc = Sidechains.extend_sidechain(0, angles, [fake_prev_c] + bb_arr, input_seq)
        return sc


def init_backbone(angles, device):
    """ Given an angle matrix (RES x ANG), this initializes the first 3 backbone points (which are arbitrary) and
        returns a TensorArray of the size required to hold all the coordinates. """
    a1 = torch.FloatTensor(torch.zeros(3, requires_grad=True).to(device))

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
    bb_pts = []
    for j in range(3):
        if j == 0:
            # we are placing N
            t = angles[i, 4]  # thetas["ca-c-n"]
            b = BONDLENS["c-n"]
            dihedral = angles[i - 1, 1]  # psi of previous residue
        elif j == 1:
            # we are placing Ca
            t = angles[i, 5]  # thetas["c-n-ca"]
            b = BONDLENS["n-ca"]
            dihedral = angles[i - 1, 2]  # omega of previous residue
        else:
            # we are placing C
            t = angles[i, 3]  # thetas["n-ca-c"]
            b = BONDLENS["ca-c"]
            dihedral = angles[i, 0]  # phi of current residue
        p3 = coords[-3]
        p2 = coords[-2]
        p1 = coords[-1]
        next_pt = nerf(p3, p2, p1, b, t, dihedral, device)
        bb_pts.append(next_pt)


    return bb_pts


def l2_normalize(t, device, eps=1e-12):
    """ Safe L2-normalization for pytorch."""
    epsilon = torch.FloatTensor([eps]).to(device)
    return t / torch.sqrt(torch.max((t**2).sum(), epsilon))


def nerf(a, b, c, l, theta, chi, device):
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

    W_hat = l2_normalize(b - a, device)
    x_hat = l2_normalize(c - b, device)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = torch.cross(W_hat, x_hat)
    z_hat = l2_normalize(n_unit, device)
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
