import torch

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
            '3c-cx': 1.526}

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
            'cx-2c-sh': 108.6}

SC_DATA = {"ARG": {"angles": ["n -cx-c8", "cx-c8-c8", "c8-c8-c8", "c8-c8-n2", "c8-n2-ca", "n2-ca-n2"],
                   "bonds": ["cx-c8", "c8-c8", "c8-c8", "c8-n2", "n2-ca", "ca-n2"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
                   "pred_atoms": ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1']},
           "HIS": {"angles": ["ct-cx-n ", "cc-ct-cx", "ct-cc-cv"],
                   "bonds": ["cx-ct", "cc-ct", "cc-cv"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
                   "pred_atoms": ['CB', 'CG', 'CD2']},
           "LYS": {"angles": ["n -cx-c8", "cx-c8-c8", "c8-c8-c8", "c8-c8-c8", "c8-c8-n3"],
                   "bonds": ["cx-c8", "c8-c8", "c8-c8", "c8-c8", "c8-n3"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
                   "pred_atoms": ['CB', 'CG', 'CD', 'CE', 'NZ']},
           "ASP": {"angles": ["n -cx-2c", "cx-2c-co", "2c-co-o2"],
                   "bonds": ["cx-2c", "2c-co", "co-o2"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
                   "pred_atoms": ['CB', 'CG', 'OD1']},
           "GLU": {"angles": ["n -cx-2c", "cx-2c-2c", "2c-2c-co", "2c-co-o2"],
                   "bonds": ["cx-2c", "2c-2c", "2c-co", "co-o2"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
                   "pred_atoms": ['CB', 'CG', 'CD', 'OE1']},
           "SER": {"angles": ["n -cx-2c", "cx-2c-oh"],
                   "bonds": ["cx-2c", "2c-oh"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'OG'],
                   "pred_atoms": ['CB', 'OG']},
           "THR": {"angles": ["n -cx-3c", "cx-3c-ct"],
                   "bonds": ["cx-3c", "3c-ct"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
                   "pred_atoms": ['CB', 'CG2']},
           "ASN": {"angles": ["n -cx-2c", "cx-2c-c ", "2c-c -o "],
                   "bonds": ["cx-2c", "2c-c ", "c -o "],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
                   "pred_atoms": ['CB', 'CG', 'OD1']},
           "GLN": {"angles": ["n -cx-2c", "cx-2c-2c", "2c-2c-c ", "2c-c -o "],
                   "bonds": ["cx-2c", "2c-2c", "2c-c ", "c -o "],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
                   "pred_atoms": ['CB', 'CG', 'CD', 'OE1']},
           "CYS": {"angles": ["n -cx-2c", "cx-2c-sh"],
                   "bonds": ["cx-2c", "sh-2c"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'SG'],
                   "pred_atoms": ['CB', 'SG']},
           "VAL": {"angles": ["n -cx-3c", "cx-3c-ct"],
                   "bonds": ["cx-3c", "3c-ct"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
                   "pred_atoms": ['CB', 'CG1']},
           "ILE": {"angles": ["n -cx-3c", "cx-3c-2c", "3c-2c-ct"],
                   "bonds": ["cx-3c", "3c-2c", "2c-ct"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
                   "pred_atoms": ['CB', 'CG1', 'CD1']},
           "LEU": {"angles": ["n -cx-2c", "cx-2c-3c", "2c-3c-ct"],
                   "bonds": ["cx-2c", "2c-3c", "3c-ct"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
                   "pred_atoms": ['CB', 'CG', 'CD1']},
           "MET": {"angles": ["n -cx-2c", "cx-2c-2c", "2c-2c-s ", "2c-s -ct"],
                   "bonds": ["cx-2c", "2c-2c", "2c-s ", "s -ct"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
                   "pred_atoms": ['CB', 'CG', 'SD', 'CE']},
           "PHE": {"angles": ["n -cx-2c", "cx-2c-ca", "2c-ca-ca"],
                   "bonds": ["cx-2c", "2c-ca", "ca-ca"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                   "pred_atoms": ['CB', 'CG', 'CD1']},
           "TYR": {"angles": ["n -cx-2c", "cx-2c-ca", "2c-ca-ca"],
                   "bonds": ["cx-2c", "2c-ca", "ca-ca"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
                   "pred_atoms": ['CB', 'CG', 'CD1']},
           "TRP": {"angles": ["ct-cx-n ", "cx-ct-c*", "ct-c*-cw"],
                   "bonds": ["cx-ct", "ct-c*", "c*-cw"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3',
                                 'CH2'],
                   "pred_atoms": ['CB', 'CG', 'CD1']},
           "GLY": {"angles": [],
                   "bonds": [],
                   "all_atoms": ['N', 'CA', 'C', 'O'],
                   "pred_atoms": []},  # no sidechain
           "PRO": {"angles": [],
                   "bonds": [],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
                   "pred_atoms": []},  # special case
           "ALA": {"angles": ["ct-cx-n "],
                   "bonds": ["cx-ct"],
                   "all_atoms": ['N', 'CA', 'C', 'O', 'CB'],
                   "pred_atoms": ['CB']},  # only has beta-carbon
           }


def extend_sidechain(i, d, bb_arr, input_seq, return_tuples=False):
    """ Given an index (i) into an angle tensor (d), builds the requested sidechain and returns it as a list."""
    residue_code = torch.argmax(input_seq[i])
    info = (i, d, bb_arr)
    codes = ["CYS","ASP","SER","GLN","LYS","ILE","PRO","THR","PHE","ASN","GLY","HIS","LEU","ARG","TRP",
             "ALA","VAL","GLU","TYR","MET"]
    return extend_any_sc(info, codes[residue_code], return_tuples)


def generate_sidechain_dihedrals(angles, i):
    """ Returns a generator that iteratively produces the sidechain dihedral angles for residue (i) in (angles). """
    first = True # for generating the first atom, cb, which depends on angle 0
    start = 6
    assert len(angles.shape) == 2 and angles.shape[1] == 11
    while start < angles.shape[-1]:
        if first:
            yield angles[i, 0]
            first = False
            continue
        if not first:
            yield angles[i, start]
            start += 1


def extend_any_sc(info, aa_code, return_tuples=False):
    """ Given a bunch of info (angle tensors, relevant bb and sc coords) and an amino acid code, generates the coords
        for that specific AA. Returns a pointer to the """
    import transformer.Structure as Structure
    lens = map(lambda bondname: BONDLENS[bondname], SC_DATA[aa_code]["bonds"])
    angs = map(lambda anglname: torch.tensor(BONDANGS[anglname]), SC_DATA[aa_code]["angles"])
    i, angles, bb_arr = info
    sc_pts = []
    dihedrals = generate_sidechain_dihedrals(angles, i)
    n2 = bb_arr[-4]  # C from the previous residue
    n1 = bb_arr[-3]  # N from cur res
    n0 = bb_arr[-2]  # Ca from cur res

    for l, a, dihe in zip(lens, angs, dihedrals):
        next_pt = Structure.nerf(n2, n1, n0, l, a, dihe,device=torch.device("cpu"))
        n2, n1, n0 = n1, n0, next_pt
        sc_pts.append(next_pt)

    if return_tuples:
        return sc_pts, aa_code, SC_DATA[aa_code]["pred_atoms"]

    return sc_pts
