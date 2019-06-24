import argparse
import datetime
import multiprocessing
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import prody as pr
import torch
import tqdm
from sklearn.model_selection import train_test_split

sys.path.extend("../protein/")
from protein.Sidechains import SC_DATA, NUM_PREDICTED_ANGLES
pr.confProDy(verbosity='error')


class IncompleteStructureError(Exception):
    """An exception to raise when a structure is incomplete."""
    def __init__(self, message):
        self.message = message


class NonStandardAminoAcidError(Exception):
    """An exception to raise when a structure contains a Non-standard amino acid."""
    def __init__(self, *args):
        super().__init__(*args)


def angle_list_to_sin_cos(angs, reshape=True):
    """ Given a list of angles, returns a new list where those angles have
        been turned into their sines and cosines. If reshape is False, a new dim.
        is added that can hold the sine and cosine of each angle,
        i.e. (len x #angs) -> (len x #angs x 2). If reshape is true, this last
        dim. is squashed so that the list of angles becomes
        [cos sin cos sin ...]. """
    new_list = []
    for a in angs:
        new_mat = np.zeros((a.shape[0], a.shape[1], 2))
        new_mat[:, :, 0] = np.cos(a)
        new_mat[:, :, 1] = np.sin(a)
        if reshape:
            new_list.append(new_mat.reshape(-1, NUM_PREDICTED_ANGLES * 2))
        else:
            new_list.append(new_mat)
    return new_list


def seq_to_onehot(seq):
    """ Given an AA sequence, returns a vector of one-hot vectors."""
    vector_array = []
    for aa in seq:
        one_hot = np.zeros(len(AA_MAP), dtype=bool)
        one_hot[AA_MAP[aa]] = 1
        vector_array.append(one_hot)
    return np.asarray(vector_array)


def get_bond_angles(res, next_res):
    """ Given 2 residues, returns the ncac, cacn, and cnca bond angles between them."""
    atoms = res.backbone.copy()
    if len(atoms) < 3:
        raise IncompleteStructureError('Missing backbone atoms.')
    atoms_next = next_res.backbone.copy()
    ncac = pr.calcAngle(atoms[0], atoms[1], atoms[2], radian=True)
    cacn = pr.calcAngle(atoms[1], atoms[2], atoms_next[0], radian=True)
    cnca = pr.calcAngle(atoms[2], atoms_next[0], atoms_next[1], radian=True)
    return ncac, cacn, cnca


def measure_bond_angles(residue, res_idx, all_res):
    """ Given a residue, measure the ncac, cacn, and cnca bond angles. """
    if res_idx == len(all_res) - 1:
        atoms = residue.backbone.copy()
        ncac = pr.calcAngle(atoms[0], atoms[1], atoms[2], radian=True)
        bondangles = [ncac, 0, 0]
    else:
        bondangles = list(get_bond_angles(residue, all_res[res_idx + 1]))
    return bondangles


def measure_phi_psi_omega(residue, outofboundchar=0):
    """ Returns phi, psi, omega for a residue, replacing out-of-bounds angles with outofboundchar."""
    try:
        phi = pr.calcPhi(residue, radian=True, dist=None)
    except ValueError:
        phi = outofboundchar
    try:
        psi = pr.calcPsi(residue, radian=True, dist=None)
    except ValueError:
        psi = outofboundchar
    try:
        omega = pr.calcOmega(residue, radian=True, dist=None)
    except ValueError:
        omega = outofboundchar
    return [phi, psi, omega]


def compute_single_dihedral(atoms):
    """ Given an iterable of 4 Atoms, uses Prody to calculate the dihedral angle between them in radians. """
    return pr.calcDihedral(atoms[0], atoms[1], atoms[2], atoms[3], radian=True)[0]


def get_dihedral(coords1, coords2, coords3, coords4, radian=False):
    """ Returns the dihedral angle in degrees. Modified from prody.measure.measure to use a numerically safe
        normalization method. """
    rad2deg = 180 / np.pi
    eps = 1e-6

    a1 = coords2 - coords1
    a2 = coords3 - coords2
    a3 = coords4 - coords3

    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1) ** 0.5
    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1) ** 0.5
    porm = np.sign((v1 * a3).sum(-1))
    arccos_input_raw = (v1 * v2).sum(-1) / ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5
    if -1 <= arccos_input_raw <= 1:
        arccos_input = arccos_input_raw
    elif arccos_input_raw > 1 and arccos_input_raw - 1 < eps:
        arccos_input = 1
    elif arccos_input_raw < -1 and np.abs(arccos_input_raw) - 1 < eps:
        arccos_input = -1
    else:
        raise ArithmeticError("Numerical issue with input to arccos.")
    rad = np.arccos(arccos_input)
    if not porm == 0:
        rad = rad * porm
    if radian:
        return rad
    else:
        return rad * rad2deg


def check_standard_continuous(residue, prev_res_num):
    """ Asserts that the residue is standard and that the chain is continuous. """
    if not residue.isstdaa:
        raise NonStandardAminoAcidError("Found a non-std AA. This should have been handled previously.")
    if residue.getResnum() != prev_res_num:
        raise IncompleteStructureError("Chain is missing residues.")
    return True


def compute_all_res_dihedrals(atom_names, residue, prev_residue, backbone, bondangles, next_res, pad_char=0):
    """ Computes all angles to predict for a given residue. If the residue is the first in the protein chain,
        a fictitious C atom is placed before the first N. This is used to compute a [C-1, N, CA, CB] dihedral
        angle. If it is not the first residue in the chain, the previous residue's C is used instead.
        Then, each group of 4 atoms in atom_names is used to generate a list of dihedral angles for this
        residue. """
    res_dihedrals = []
    if len(atom_names) > 0:
        if prev_residue is None:
            atoms = [residue.select("name " + an) for an in atom_names]

            try:
                res_dihedrals = [get_dihedral(next_res.select("name N").getCoords()[0],
                                              residue.select("name C").getCoords()[0],
                                              residue.select("name CA").getCoords()[0],
                                              residue.select("name CB").getCoords()[0], radian=True)]
            except AttributeError:
                raise IncompleteStructureError(f'Mising atoms at start of residue {residue} or {next_res}.')
        elif prev_residue is not None:
            atoms = [prev_residue.select("name C")] + [residue.select("name " + an) for an in atom_names]

        if (prev_residue is not None and len(atoms) != len(atom_names) + 1) \
            or (prev_residue is None and len(atoms) != len(atom_names)) or None in atoms:
            raise IncompleteStructureError(f'Missing atoms in residue {residue}.')
        for n in range(len(atoms) - 3):
            dihe_atoms = atoms[n:n + 4]
            res_dihedrals.append(compute_single_dihedral(dihe_atoms))
    resname = residue.getResname()
    if resname not in ["LEU", "ILE", "VAL", "THR"]:
        return backbone + bondangles + res_dihedrals + (NUM_PREDICTED_ANGLES - 6 - len(res_dihedrals)) * [pad_char]
    if resname == "LEU":
        first_three = ["CA", "CB", "CG"]
        next_atom = "CD2"
    elif resname == "ILE":
        first_three = ["N", "CA", "CB"]
        next_atom = "CG2"
    elif resname == "VAL":
        first_three = ["N", "CA", "CB"]
        next_atom = "CG2"
    elif resname == "THR":
        first_three = ["N", "CA", "CB"]
        next_atom = "OG1"
    atom_selections = [residue.select("name " + an) for an in first_three + [next_atom]]
    if len(atom_selections) != 4 or None in atom_selections:
        raise IncompleteStructureError('Missing sidechain atoms.')
    res_dihedrals.append(compute_single_dihedral(atom_selections))
    assert len(res_dihedrals) + len(backbone + bondangles) == 10 and resname in ["ILE", "LEU"] or len(res_dihedrals) + \
           len(backbone + bondangles) == 9 and resname in ["VAL", "THR"], \
        "Angle position in array must match what is assumed in Sidechains:extend_any_sidechain."

    return backbone + bondangles + res_dihedrals + (NUM_PREDICTED_ANGLES - 6 - len(res_dihedrals)) * [pad_char]


# get angles from chain
def get_angles_from_chain(chain):
    """ Given a ProDy Chain object (from a Hierarchical View), return a numpy array of
        angles. Returns None if the PDB should be ignored due to weird artifacts. Also measures
        the bond angles along the peptide backbone, since they account for significat variation.
        i.e. [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2, chi3, chi4, chi5], [...] ...] """

    dihedrals = []
    if chain.nonstdaa:
        raise NonStandardAminoAcidError
    sequence = chain.getSequence()
    chain = chain.select("protein and not hetero").copy()
    # TODO remove references to previous residue - not necessary, instead use next residue
    all_residues = list(chain.iterResidues())
    prev = all_residues[0].getResnum()
    prev_res = None
    for res_id, res in enumerate(all_residues):
        check_standard_continuous(res, prev)
        if len(res.backbone) < 3:
            raise IncompleteStructureError(f"Incomplete backbone for residue {res}.")
        prev = res.getResnum() + 1

        res_backbone = measure_phi_psi_omega(res)
        res_bond_angles = measure_bond_angles(res, res_id, all_residues)

        atom_names = ["N", "CA"]
        # TODO verify correctness of GLY, PRO atom_names
        if res.getResname() is "GLY":
            atom_names = SC_DATA[res.getResname()]["predicted"]
        else:
            atom_names += SC_DATA[res.getResname()]["predicted"]
        if res_id == 0:
            next_res = all_residues[1]
        else:
            next_res = None
        calculated_dihedrals = compute_all_res_dihedrals(atom_names, res, prev_res, res_backbone, res_bond_angles,
                                                         next_res)
        dihedrals.append(calculated_dihedrals)
        prev_res = res

    dihedrals_np = np.asarray(dihedrals)
    assert not np.any(np.isnan(dihedrals_np)), "NaNs are present in the dihedral array."
    return dihedrals_np, sequence


def work(pdbid_chain):
    """
    For a single PDB ID with chain, i.e. ('1A9U_A'), fetches that PDB chain from the PDB and computes its angles.
    """
    pdbid, chid = pdbid_chain.split("_")
    try:
        pdb_hv = pr.parsePDB(pdbid, chain=chid).getHierView()
    except AttributeError:
        print("Error parsing", pdbid_chain)
        ERROR_FILE.write(f"{pdbid_chain}\n")
        return None
    assert pdb_hv.numChains() == 1, "Only a single chain should be parsed from the PDB."
    chain = pdb_hv[chid]
    assert chain.getChid() == chid, "The chain ID was not as expected."
    try:
        dihedrals_sequence = get_angles_from_chain(chain)
    except (IncompleteStructureError, NonStandardAminoAcidError):
        return None

    dihedrals, sequence = dihedrals_sequence

    return dihedrals, sequence, pdbid_chain


def additional_checks(matrix):
    """ Returns true if a matrix contains NaNs, infs, or all 0s."""
    zeros = not np.any(matrix)
    if not np.any(np.isnan(matrix)) and not np.any(np.isinf(matrix)) and not zeros:
        return True
    else:
        return False


def unpack_processed_results(results):
    """
    Given an iterable of processed results containing angles, sequences, and PDB IDs, this function separates out
    the components (sequences as one-hot vectors, angle matrices, and PDB IDs) iff all were successfully preprocessed.
    """
    all_ohs = []
    all_angs = []
    all_ids = []
    c = 0
    for r in results:
        if not r:
            # PDB failed to download
            continue
        ang, seq, i = r
        if len(seq[0]) > args.max_len:
            continue
        for j in range(len(ang)):
            a, oh, _i = ang[j], seq_to_onehot(seq[j]), i[j]
            if additional_checks(oh) and additional_checks(a):
                all_ohs.append(oh)
                all_angs.append(a)
                all_ids.append(_i)
                c += 1
    print(c, "chains successfully parsed and downloaded.")
    return all_ohs, all_angs, all_ids


def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Given split data along with the query information that generated it, this function saves the data as a Python
    dictionary, which is then saved to disk using torch.save.
    """
    # Separate PDB ID/Sequence tuples.
    X_train_ids = [x[1] for x in X_train]
    X_test_ids = [x[1] for x in X_test]
    X_val_ids = [x[1] for x in X_val]
    X_train = [x[0] for x in X_train]
    X_test = [x[0] for x in X_test]
    X_val = [x[0] for x in X_val]

    train_proteinnet_dict = torch.load(os.path.join(args.input_dir, TRAIN_FILE))
    valid_proteinnet_dict = torch.load(os.path.join(args.input_dir, "validation.pt"))
    test_proteinnet_dict = torch.load(os.path.join(args.input_dir, "testing.pt"))

    # Create a dictionary data structure, using the sin/cos transformed angles
    date = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    data = {"train": {"seq": X_train,
                      "ang": angle_list_to_sin_cos(y_train),
                      "ids": X_train_ids,
                      "mask": [train_proteinnet_dict[_id]["mask"] for _id in X_train_ids],
                      "evolutionary": [train_proteinnet_dict[_id]["evolutionary"] for _id in X_train_ids],
                      "secondary": [train_proteinnet_dict[_id]["secondary"] for _id in X_train_ids]},
            "valid": {"seq": X_val,
                      "ang": angle_list_to_sin_cos(y_val),
                      "ids": X_val_ids,
                      "mask": [valid_proteinnet_dict[_id]["mask"] for _id in X_val_ids],
                      "evolutionary": [valid_proteinnet_dict[_id]["evolutionary"] for _id in X_val_ids],
                      "secondary": [valid_proteinnet_dict[_id]["secondary"] for _id in X_val_ids]},
            "test": {"seq": X_test,
                     "ang": angle_list_to_sin_cos(y_test),
                     "ids": X_test_ids,
                     "mask": [test_proteinnet_dict[_id]["mask"] for _id in X_test_ids],
                     "evolutionary": [test_proteinnet_dict[_id]["evolutionary"] for _id in X_test_ids],
                     "secondary": [test_proteinnet_dict[_id]["secondary"] for _id in X_test_ids]},
            "settings": {"max_len": max(map(len, X_train + X_val + X_test))},
            "description": {"ProteinNet"},  # TODO add more informative description
            "query": "ProteinNet",
            "date": {date}}
    # To parse date later, use datetime.datetime.strptime(date, "%I:%M%p on %B %d, %Y")

    if args.pickle:
        with open(args.out_file, "wb") as f:
            pickle.dump(data, f)
    else:
        torch.save(data, args.out_file)


def split_data(all_ohs, all_angs, all_ids):
    """
    Given the entire dataset as 3 lists (one-hot sequences, angles, and IDs), this functions splits it into training,
    validation, and testing sets.

    If the script is run in ProteinNet mode, the script will follow ProteinNet's guidelines for splits.
    """
    ohs_ids = list(zip(all_ohs, all_ids))
    X_train, X_test, y_train, y_test = train_test_split(ohs_ids, all_angs, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
    print("Train, test, validation set sizes:\n" + str(list(map(len, [X_train, X_test, X_val]))))
    return X_train, X_val, X_test, y_train, y_val, y_test


def fetch_pdb_ids(filename):
    """
    Given a ProteinNet file, loads the query + description and performs a fetch for the relevant PDB IDs.
    """
    pdb_ids = []
    pn_file = os.path.join(args.input_dir, filename)
    text_file = pn_file.replace('.pt', '.ids')
    if os.path.exists(text_file):
        with open(text_file, "r") as f:
            line = f.readline().strip()
            while line != "":
                pdb_ids.append(line)
                line = f.readline().strip()
    else:
        text_file_open = open(text_file, "w")
        d = torch.load(pn_file)
        for pnid, data in d.items():
            try:
                pdb_id, model_id, chain_id = pnid.split("_")
            except ValueError:
                # TODO Implement support for ASTRAL data
                # ASTRAL data points have only "PDBID_domain" and are currently ignored
                continue
            pdb_ids.append(f"{pdb_id}_{chain_id}")
            text_file_open.write(f"{pdb_id}_{chain_id}\n")
        text_file_open.close()

    return pdb_ids


def main():
    # Load query and fetch PDB IDs
    train_pdb_ids = fetch_pdb_ids(TRAIN_FILE)
    # validation_pdb_ids = fetch_pdb_ids("validation.pt")

    # Download and preprocess all data from PDB IDs
    with Pool(multiprocessing.cpu_count()) as p:
        results = list(tqdm.tqdm(p.imap(work, train_pdb_ids), total=len(train_pdb_ids)))
    ERROR_FILE.close()

    # Unpack results, throwing out malformed data
    all_ohs, all_angs, all_ids = unpack_processed_results(results)

    # Split into train, test and validation sets. Report sizes.
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(all_ohs, all_angs, all_ids)
    save_data(x_train, x_val, x_test, y_train, y_val, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a ProteinNet directory of raw records into a dataset for all"
                                                 "atom protein structure prediction.")
    parser.add_argument('input_dir', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('-o', '--out_file', type=str, help='Path to output file (.tch file)')
    parser.add_argument("--pdb_dir", default="/home/jok120/pdb/", type=str, help="Path for ProDy-downloaded PDB files.")
    parser.add_argument("-p", "--pickle", action="store_true",
                        help="Save data as a pickled dictionary instead of a torch-dictionary.")
    args = parser.parse_args()
    TRAIN_FILE = "training_100.pt"
    AA_MAP = {'A': 0,  'C': 1,  'D': 2,  'E': 3,
              'F': 4,  'G': 5,  'H': 6,  'I': 7,
              'K': 8,  'L': 9,  'M': 10, 'N': 11,
              'P': 12, 'Q': 13, 'R': 14, 'S': 15,
              'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    pr.pathPDBFolder(args.pdb_dir)
    np.set_printoptions(suppress=True)  # suppresses scientific notation when printing
    np.set_printoptions(threshold=np.nan)  # suppresses '...' when printing
    today = datetime.datetime.today()
    suffix = today.strftime("%y%m%d")
    if not args.out_file and args.pickle:
        args.out_file = "../data/proteinnet/" + suffix + ".pkl"
    elif not args.out_file and not args.pickle:
        args.out_file = "../data/proteinnet/" + suffix + ".pt"
    ERROR_FILE = open("error.log", "w")
    main()

