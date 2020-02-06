import numpy as np
import torch

NUM_PREDICTED_ANGLES = 13
NUM_BB_TORSION_ANGLES = 3
NUM_BB_OTHER_ANGLES = 3
NUM_SC_ANGLES = NUM_PREDICTED_ANGLES - (NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES)
NUM_PREDICTED_COORDS = 13

ONE_TO_THREE_LETTER_MAP = {"R": "ARG", "H": "HIS", "K": "LYS", "D": "ASP", "E": "GLU", "S": "SER", "T": "THR",
                           "N": "ASN", "Q": "GLN", "C": "CYS", "G": "GLY", "P": "PRO", "A": "ALA", "V": "VAL",
                           "I": "ILE", "L": "LEU", "M": "MET", "F": "PHE", "Y": "TYR", "W": "TRP"}

THREE_TO_ONE_LETTER_MAP = {v: k for k, v in ONE_TO_THREE_LETTER_MAP.items()}

AA_MAP = {'A': 0, 'C': 1, 'D': 2, 'E': 3,
          'F': 4, 'G': 5, 'H': 6, 'I': 7,
          'K': 8, 'L': 9, 'M': 10, 'N': 11,
          'P': 12, 'Q': 13, 'R': 14, 'S': 15,
          'T': 16, 'V': 17, 'W': 18, 'Y': 19}

for one_letter_code in list(AA_MAP.keys()):
    AA_MAP[ONE_TO_THREE_LETTER_MAP[one_letter_code]] = AA_MAP[one_letter_code]

AA_MAP_INV = {v: k for k, v in AA_MAP.items()}


def deg2rad(angle):
    """
    Converts an angle in degrees to radians.
    """
    return angle * np.pi / 180.


def extend_sidechain(i, d, bb_arr, input_seq, return_tuples=False, first_sc=False):
    """
    Given an index (i) into an angle tensor (d), builds the requested
    """
    residue_code = torch.argmax(input_seq[i])
    info = (i, d, bb_arr)

    return extend_any_sc(info, AA_MAP_INV[residue_code], return_tuples, first_sc)


def generate_sidechain_dihedrals(angles, i):
    """
    Returns a generator that iteratively produces the sidechain dihedral
    angles for residue (i) in (angles).
    """
    assert len(angles.shape) == 2 and angles.shape[1] == NUM_PREDICTED_ANGLES, print("Improper shape for angles:", angles.shape)
    angle_idx = 6
    while angle_idx < angles.shape[-1]:
        yield angles[i, angle_idx]
        angle_idx += 1


def extend_any_sc(info, aa_code, return_tuples=False, first_sc=False):
    """
    Given a bunch of info (angle tensors, relevant bb and sc coords) and an
    amino acid code, generates the coords for that specific AA.
    """
    # TODO: clarify behavior with first sidechain. Here, it must reorganize its input to carefully place the first CB
    from .Structure import nerf
    lens = map(lambda bondname: BONDLENS[bondname], SC_DATA[aa_code]["bonds"])
    angs = map(lambda anglname: torch.tensor(deg2rad(BONDANGS[anglname])), SC_DATA[aa_code]["angles"])
    i, angles, bb_arr = info
    sc_pts = []
    dihedrals = generate_sidechain_dihedrals(angles, i)
    n2 = bb_arr[-4]  # C from the previous residue, N from next res
    n1 = bb_arr[-3]  # N from cur res, C from cur res
    n0 = bb_arr[-2]  # Ca from cur res, Ca from cur
    swap = True
    for l, a, dihe in zip(lens, angs, dihedrals):
        next_pt = nerf(n2, n1, n0, l, a, dihe)  # CB
        n2, n1, n0 = n1, n0, next_pt  # N, CA, CB
        sc_pts.append(next_pt)
        if first_sc and swap:
            n2, n1, n0 = bb_arr[-1], bb_arr[-2], next_pt
            swap = False

    # The following residues have extra *atoms* that are predicted manually
    # TODO clean up cases for the first SC, clarify ordering of bb_arr, which does not always refer to the right atoms
    if aa_code == "ILE":
        # nerf, N, CA, CB, *CG2*
        if first_sc:
            n = bb_arr[-1]
            ca = bb_arr[-2]
            cb = sc_pts[0]
        else:
            n = bb_arr[-3]
            ca = bb_arr[-2]
            cb = sc_pts[0]
        new_pt = nerf(n, ca, cb,
                                BONDLENS["ct-3c"], torch.tensor(np.deg2rad(BONDANGS['cx-3c-ct'])),
                                angles[i, 9], device=torch.device("cpu"))
        sc_pts.append(new_pt)
    if aa_code == "LEU":
        # nerf, CA, CB, CG, *CD2*
        if first_sc:
            ca = bb_arr[-2]
            cb = sc_pts[0]
            cg = sc_pts[1]
        else:
            ca = bb_arr[-2]
            cb = sc_pts[0]
            cg = sc_pts[1]
        new_pt = nerf(ca, cb, cg,
                                BONDLENS["ct-3c"], torch.tensor(np.deg2rad(BONDANGS['2c-3c-ct'])),
                                angles[i, 9], device=torch.device("cpu"))
        sc_pts.append(new_pt)
    if aa_code == "THR":
        # nerf, N, CA, CB, *OG1*
        if first_sc:
            n = bb_arr[-1]
            ca = bb_arr[-2]
            cb = sc_pts[0]
        else:
            n = bb_arr[-3]
            ca = bb_arr[-2]
            cb = sc_pts[0]
        new_pt = nerf(n, ca, cb,
                                BONDLENS["3c-oh"], torch.tensor(np.deg2rad(BONDANGS['cx-3c-oh'])),
                                angles[i, 8], device=torch.device("cpu"))
        sc_pts.append(new_pt)
    if aa_code == "VAL":
        # nerf, N, CA, CB, *CG2*
        if first_sc:
            n = bb_arr[-1]
            ca = bb_arr[-2]
            cb = sc_pts[0]
        else:
            n = bb_arr[-3]
            ca = bb_arr[-2]
            cb = sc_pts[0]
        new_pt = nerf(n, ca, cb,
                                BONDLENS["ct-3c"], torch.tensor(np.deg2rad(BONDANGS['cx-3c-ct'])),
                                angles[i, 8], device=torch.device("cpu"))
        sc_pts.append(new_pt)

    if return_tuples:
        special_case_extra_atoms = {"LEU": "CD2", "ILE": 'CG2', 'THR': 'OG1', 'VAL': 'CG2'}
        if aa_code in special_case_extra_atoms.keys():
            predicted = SC_DATA[aa_code]["predicted"] + [special_case_extra_atoms[aa_code]]
        else:
            predicted = SC_DATA[aa_code]["predicted"]
        return sc_pts, aa_code, predicted

    return sc_pts
