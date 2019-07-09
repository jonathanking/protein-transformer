import numpy as np
import prody as pr

from protein.Sidechains import NUM_PREDICTED_ANGLES, SC_DATA
from structure_exceptions import NonStandardAminoAcidError, IncompleteStructureError


def angle_list_to_sin_cos(angs, reshape=True):
    """ Given a list of angles, returns a new list where those angles have been turned into
    their sines and cosines. If reshape is False, a new dim. is added that can hold the sine and
    cosine of each angle, i.e. (len x #angs) -> ( len x #angs x 2). If reshape is true,
    this last dim. is squashed so that the list of angles becomes [cos sin cos sin ...]. """
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


def check_standard_continuous(residue, prev_res_num):
    """ Asserts that the residue is standard and that the chain is continuous. """
    if not residue.isstdaa:
        raise NonStandardAminoAcidError("Found a non-std AA.")
    if residue.getResnum() != prev_res_num:
        raise IncompleteStructureError("Chain is missing residues.")
    return True


def compute_all_res_dihedrals(atom_names, residue, prev_residue, backbone, bondangles, next_res,
                              pad_char=0):
    """
    Computes all angles to predict for a given residue. If the residue is the first in the
    protein chain, a fictitious C atom is placed before the first N. This is used to compute a [
    C-1, N, CA, CB] dihedral angle. If it is not the first residue in the chain, the previous
    residue's C is used instead. Then, each group of 4 atoms in atom_names is used to generate a
    list of dihedral angles for this residue.
    """
    res_dihedrals = []
    if len(atom_names) > 0:
        if prev_residue is None:
            atoms = [residue.select("name " + an) for an in atom_names]

            try:
                res_dihedrals = [get_dihedral(next_res.select("name N").getCoords()[0],
                                              residue.select("name C").getCoords()[0],
                                              residue.select("name CA").getCoords()[0],
                                              residue.select("name CB").getCoords()[0],
                                              radian=True)]
            except AttributeError:
                raise IncompleteStructureError(
                    f'Mising atoms at start of residue {residue} or {next_res}.')
        elif prev_residue is not None:
            atoms = [prev_residue.select("name C")] + [residue.select("name " + an) for an in
                                                       atom_names]

        if (prev_residue is not None and len(atoms) != len(atom_names) + 1) \
                or (prev_residue is None and len(atoms) != len(atom_names)) or None in atoms:
            raise IncompleteStructureError(f'Missing atoms in residue {residue}.')
        for n in range(len(atoms) - 3):
            dihe_atoms = atoms[n:n + 4]
            res_dihedrals.append(compute_single_dihedral(dihe_atoms))
    resname = residue.getResname()
    if resname not in ["LEU", "ILE", "VAL", "THR"]:
        return backbone + bondangles + res_dihedrals + (
                    NUM_PREDICTED_ANGLES - 6 - len(res_dihedrals)) * [pad_char]
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
    assert len(res_dihedrals) + len(backbone + bondangles) == 10 and resname in ["ILE",
                                                                                 "LEU"] or len(
        res_dihedrals) + \
           len(backbone + bondangles) == 9 and resname in ["VAL", "THR"], \
        "Angle position in array must match what is assumed in Sidechains:extend_any_sidechain."

    return backbone + bondangles + res_dihedrals + (
                NUM_PREDICTED_ANGLES - 6 - len(res_dihedrals)) * [pad_char]


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
        calculated_dihedrals = compute_all_res_dihedrals(atom_names, res, prev_res, res_backbone,
                                                         res_bond_angles, next_res)
        dihedrals.append(calculated_dihedrals)
        prev_res = res

    dihedrals_np = np.asarray(dihedrals)
    return dihedrals_np, sequence


def get_angles_and_coords_from_chain(chain):
    """
    Given a ProDy Chain object (from a Hierarchical View), return a tuple (angles, coords).
    Returns None if the PDB should be ignored due to weird artifacts. Also measures the
    bond angles along the peptide backbone, since they account for significant variation.
    i.e. [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2, chi3, chi4, chi5], [...] ...]
    """
    coords = []
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
        calculated_dihedrals = compute_all_res_dihedrals(atom_names, res, prev_res, res_backbone,
                                                         res_bond_angles, next_res)

        bbcoords = [res.select(f"name {n}").getCoords()[0] for n in ["N", "CA", "C"]]
        sccoords = [res.select(f"name {n}").getCoords()[0] for n in set(atom_names) - {"N", "CA", "C", "H"}]
        rescoords = np.concatenate((np.stack(bbcoords + sccoords), np.zeros((10 - len(sccoords), 3))))

        coords.append(rescoords)
        dihedrals.append(calculated_dihedrals)
        prev_res = res

    dihedrals_np = np.asarray(dihedrals)
    coords_np = np.concatenate(coords)
    assert coords_np.shape[0] == len(sequence) * 13, f"Coords shape {coords_np.shape} does not match len(seq)*13 = {len(sequence) * 13}"
    return dihedrals_np, coords_np, sequence


def additional_checks(matrix):
    """ Returns true if a matrix does not contain NaNs, infs, or all 0s."""
    return not np.any(np.isnan(matrix)) and not np.any(np.isinf(matrix)) and np.any(matrix)


def zero_runs(arr):
    """
    Returns indices of zero-runs.
    Taken from https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array.
    >>> import numpy as np
    >>> zero_runs(np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]))
    array([[0, 4],
          [11, 13]])
    """
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(arr, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def get_bond_angles(res, next_res):
    """ Given 2 residues, returns the ncac, cacn, and cnca bond angles between them."""
    atoms = res.backbone.copy()
    atoms_next = next_res.backbone.copy()
    if len(atoms) < 3 or len(atoms_next) < 3:
        raise IncompleteStructureError('Missing backbone atoms.')
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
    """
    Returns phi, psi, omega for a residue, replacing out-of-bounds angles with
    outofboundchar.
    """
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
    """
    Given an iterable of 4 Atoms, uses Prody to calculate the dihedral angle between them in
    radians.
    """
    atoms = [a.getCoords()[0] for a in atoms]
    return get_dihedral(atoms[0], atoms[1], atoms[2], atoms[3], radian=True)


def get_dihedral(coords1, coords2, coords3, coords4, radian=False):
    """
    Returns the dihedral angle in degrees. Modified from prody.measure.measure to use a
    numerically safe normalization method.
    """
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


AA_MAP = {'A': 0, 'C': 1, 'D': 2, 'E': 3,
          'F': 4, 'G': 5, 'H': 6, 'I': 7,
          'K': 8, 'L': 9, 'M': 10, 'N': 11,
          'P': 12, 'Q': 13, 'R': 14, 'S': 15,
          'T': 16, 'V': 17, 'W': 18, 'Y': 19}