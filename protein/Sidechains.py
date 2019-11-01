import numpy as np
import torch

NUM_PREDICTED_ANGLES = 12
NUM_PREDICTED_COORDS = 13

ONE_TO_THREE_LETTER_MAP = {"R": "ARG", "H": "HIS", "K": "LYS", "D": "ASP", "E": "GLU", "S": "SER", "T": "THR",
                           "N": "ASN", "Q": "GLN", "C": "CYS", "G": "GLY", "P": "PRO", "A": "ALA", "V": "VAL",
                           "I": "ILE", "L": "LEU", "M": "MET", "F": "PHE", "Y": "TYR", "W": "TRP"}

THREE_TO_ONE_LETTER_MAP = {v: k for k, v in ONE_TO_THREE_LETTER_MAP.items()}

BONDLENS = {'cc-cv': 1.375,
            'c8-cx': 1.526,
            '2c-ca': 1.51,
            'cx-3c': 1.526,
            'c -2c': 1.522,
            '2c-3c': 1.526,
            'cx-2c': 1.526,
            'n2-ca': 1.303,
            'o -c ': 1.218,
            'n2-c8': 1.463,
            'c*-cw': 1.352,
            'ca-2c': 1.51,
            'ca-n2': 1.303,
            '3c-2c': 1.526,
            's -ct': 1.81,
            '2c-oh': 1.41,
            'cv-cc': 1.375,
            'o2-co': 1.25,
            '2c-c ': 1.522,
            'oh-2c': 1.41,
            '2c-2c': 1.526,
            'c8-n3': 1.471,
            '2c-ct': 1.526,
            'c8-n2': 1.463,
            'c -o ': 1.218,
            'co-2c': 1.522,
            'n3-c8': 1.471,
            'ct-s ': 1.81,
            'sh-2c': 1.81,
            '2c-cx': 1.526,
            'co-o2': 1.25,
            'cx-c8': 1.526,
            '2c-sh': 1.81,
            'ct-cc': 1.504,
            's -2c': 1.81,
            'cx-ct': 1.526,
            'ct-2c': 1.526,
            'ct-3c': 1.526,
            'ct-c*': 1.495,
            '2c-s ': 1.81,
            '3c-ct': 1.526,
            'cc-ct': 1.504,
            'c8-c8': 1.526,
            'ca-ca': 1.398,
            'c*-ct': 1.495,
            '2c-co': 1.522,
            'cw-c*': 1.352,
            'ct-cx': 1.526,
            '3c-cx': 1.526,
            '3c-oh': 1.4218,  # measured from PyMOL's build feature for THR
            'cx-2c-proline': 1.5361,  # measured from PyMOL for PRO
            }

BONDANGS = {'c8-cx-n ': 109.7,
            'cx-2c-c ': 111.1,
            'o -c -2c': 120.4,
            'c -2c-2c': 111.1,
            '2c-2c-c ': 111.1,
            'c8-c8-cx': 109.5,
            '2c-2c-co': 111.1,
            'n -cx-3c': 109.7,
            '2c-s -ct': 98.9,
            'ct-2c-3c': 109.5,
            'cx-2c-2c': 109.5,
            'cw-c*-ct': 125.0,
            '2c-c -o ': 120.4,
            'cc-ct-cx': 113.1,
            '3c-2c-cx': 109.5,
            'ct-cc-cv': 120.0,
            '2c-3c-cx': 109.5,
            'ct-c*-cw': 125.0,
            '2c-3c-ct': 109.5,
            'c8-c8-n3': 111.2,
            'cx-3c-2c': 109.5,
            '3c-cx-n ': 109.7,
            'ct-3c-cx': 109.5,
            'cx-2c-co': 111.1,
            '2c-cx-n ': 109.7,
            'cx-c8-c8': 109.5,
            'n -cx-ct': 109.7,
            'cx-3c-ct': 109.5,
            'c8-c8-c8': 109.5,
            '2c-co-o2': 117.0,
            'cx-ct-c*': 115.6,
            '2c-ca-ca': 120.0,
            'ct-3c-2c': 109.5,
            'n2-c8-c8': 111.2,
            'c8-c8-n2': 111.2,
            '2c-2c-cx': 109.5,
            'co-2c-cx': 111.1,
            'cx-2c-oh': 109.5,
            'ca-ca-2c': 120.0,
            'cx-2c-3c': 109.5,
            'sh-2c-cx': 108.6,
            'cx-2c-ca': 114.0,
            '2c-2c-s ': 114.7,
            'c -2c-cx': 111.1,
            'ct-s -2c': 98.9,
            'n -cx-2c': 109.7,
            'ca-n2-c8': 123.2,
            'o2-co-2c': 117.0,
            'n -cx-c8': 109.7,
            'c8-n2-ca': 123.2,
            'n3-c8-c8': 111.2,
            'n2-ca-n2': 120.0,
            'c*-ct-cx': 115.6,
            'cv-cc-ct': 120.0,
            'ca-2c-cx': 114.0,
            'ct-cx-n ': 109.7,
            'co-2c-2c': 111.1,
            'cx-ct-cc': 113.1,
            '3c-2c-ct': 109.5,
            's -2c-2c': 114.7,
            'oh-2c-cx': 109.5,
            'cx-2c-sh': 108.6,
            'cx-3c-oh': 110.6,  # measured from PyMOL's build feature for THR
            'n -cx-2c-proline': 101.88  # measured from PyMOL for PRO
            }

SC_DATA = {"ARG": {"angles": ["n -cx-c8", "cx-c8-c8", "c8-c8-c8", "c8-c8-n2", "c8-n2-ca", "n2-ca-n2"],
                   "bonds": ["cx-c8", "c8-c8", "c8-c8", "c8-n2", "n2-ca", "ca-n2"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
                   "predicted": ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1']},
           "HIS": {"angles": ["ct-cx-n ", "cc-ct-cx", "ct-cc-cv"],
                   "bonds": ["cx-ct", "cc-ct", "cc-cv"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
                   "predicted": ['CB', 'CG', 'CD2']},
           "LYS": {"angles": ["n -cx-c8", "cx-c8-c8", "c8-c8-c8", "c8-c8-c8", "c8-c8-n3"],
                   "bonds": ["cx-c8", "c8-c8", "c8-c8", "c8-c8", "c8-n3"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
                   "predicted": ['CB', 'CG', 'CD', 'CE', 'NZ']},
           "ASP": {"angles": ["n -cx-2c", "cx-2c-co", "2c-co-o2"],
                   "bonds": ["cx-2c", "2c-co", "co-o2"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
                   "predicted": ['CB', 'CG', 'OD1']},
           "GLU": {"angles": ["n -cx-2c", "cx-2c-2c", "2c-2c-co", "2c-co-o2"],
                   "bonds": ["cx-2c", "2c-2c", "2c-co", "co-o2"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
                   "predicted": ['CB', 'CG', 'CD', 'OE1']},
           "SER": {"angles": ["n -cx-2c", "cx-2c-oh"],
                   "bonds": ["cx-2c", "2c-oh"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'OG'],
                   "predicted": ['CB', 'OG']},
           "THR": {"angles": ["n -cx-3c", "cx-3c-ct"],
                   "bonds": ["cx-3c", "3c-ct"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
                   "predicted": ['CB', 'CG2']},
           "ASN": {"angles": ["n -cx-2c", "cx-2c-c ", "2c-c -o "],
                   "bonds": ["cx-2c", "2c-c ", "c -o "],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
                   "predicted": ['CB', 'CG', 'OD1']},
           "GLN": {"angles": ["n -cx-2c", "cx-2c-2c", "2c-2c-c ", "2c-c -o "],
                   "bonds": ["cx-2c", "2c-2c", "2c-c ", "c -o "],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
                   "predicted": ['CB', 'CG', 'CD', 'OE1']},
           "CYS": {"angles": ["n -cx-2c", "cx-2c-sh"],
                   "bonds": ["cx-2c", "sh-2c"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'SG'],
                   "predicted": ['CB', 'SG']},
           "VAL": {"angles": ["n -cx-3c", "cx-3c-ct"],
                   "bonds": ["cx-3c", "3c-ct"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
                   "predicted": ['CB', 'CG1']},
           "ILE": {"angles": ["n -cx-3c", "cx-3c-2c", "3c-2c-ct"],
                   "bonds": ["cx-3c", "3c-2c", "2c-ct"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
                   "predicted": ['CB', 'CG1', 'CD1']},
           "LEU": {"angles": ["n -cx-2c", "cx-2c-3c", "2c-3c-ct"],
                   "bonds": ["cx-2c", "2c-3c", "3c-ct"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
                   "predicted": ['CB', 'CG', 'CD1']},
           "MET": {"angles": ["n -cx-2c", "cx-2c-2c", "2c-2c-s ", "2c-s -ct"],
                   "bonds": ["cx-2c", "2c-2c", "2c-s ", "s -ct"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
                   "predicted": ['CB', 'CG', 'SD', 'CE']},
           "PHE": {"angles": ["n -cx-2c", "cx-2c-ca", "2c-ca-ca"],
                   "bonds": ["cx-2c", "2c-ca", "ca-ca"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                   "predicted": ['CB', 'CG', 'CD1']},
           "TYR": {"angles": ["n -cx-2c", "cx-2c-ca", "2c-ca-ca"],
                   "bonds": ["cx-2c", "2c-ca", "ca-ca"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
                   "predicted": ['CB', 'CG', 'CD1']},
           "TRP": {"angles": ["ct-cx-n ", "cx-ct-c*", "ct-c*-cw"],
                   "bonds": ["cx-ct", "ct-c*", "c*-cw"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
                   "predicted": ['CB', 'CG', 'CD1']},
           "GLY": {"angles": [],
                   "bonds": [],
                   "all": ['N', 'CA', 'C', 'O'],
                   "predicted": []},  # no sidechain
           "PRO": {"angles": ["n -cx-2c-proline"],
                   "bonds": ["cx-2c-proline"],
                   "all": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
                   "predicted": ['CB']},  # special case
           "ALA": {"angles": ["ct-cx-n "],
                   "bonds": ["cx-ct"],
                   "all": ['N', 'CA', 'C', 'O', 'CB'],
                   "predicted": ['CB']},  # only has beta-carbon
           }

for res in SC_DATA.keys():
    # Missing atoms are those not in the backbone and not predicted
    SC_DATA[res]["missing"] = list(set(SC_DATA[res]["all"])
                                   - {"N", "CA", "C", "O"}
                                   - set(SC_DATA[res]["predicted"]))

    # Align target atoms are the last 3 predicted, used to align the mobile part of the sidechain to be constructed
    SC_DATA[res]["align_target"] = SC_DATA[res]["predicted"][-3:] if len(SC_DATA[res]["predicted"]) >= 3 else []
    if res == "PRO":
        SC_DATA[res]["align_target"] = ["N", "CA", "CB"]
    elif res == "THR":
        SC_DATA[res]["align_target"] = ["CA", "CB", "CG2"]
    elif res == "VAL":
        SC_DATA[res]["align_target"] = ["CA", "CB", "CG1"]

    # Align mobile atoms are the total set of atoms that will be aligned, includes missing and target atoms
    SC_DATA[res]["align_mobile"] = list(set(SC_DATA[res]["align_target"] + SC_DATA[res]["missing"]))


def deg2rad(angle):
    """
    Converts an angle in degrees to radians.
    """
    return angle * np.pi / 180.


def extend_sidechain(i, d, bb_arr, input_seq, return_tuples=False, first_sc=False):
    """
    Given an index (i) into an angle tensor (d), builds the requested
    sidechain and returns it as a list.
    """
    residue_code = torch.argmax(input_seq[i])
    info = (i, d, bb_arr)
    codes = ["CYS","ASP","SER","GLN","LYS","ILE","PRO","THR","PHE","ASN","GLY","HIS","LEU","ARG","TRP",
             "ALA","VAL","GLU","TYR","MET"]
    return extend_any_sc(info, codes[residue_code], return_tuples, first_sc)


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
    import protein.Structure as Structure
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
        next_pt = Structure.nerf(n2, n1, n0, l, a, dihe)  # CB
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
        new_pt = Structure.nerf(n, ca, cb,
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
        new_pt = Structure.nerf(ca, cb, cg,
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
        new_pt = Structure.nerf(n, ca, cb,
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
        new_pt = Structure.nerf(n, ca, cb,
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
