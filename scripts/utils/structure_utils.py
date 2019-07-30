import numpy as np
import prody as pr

from protein.Sidechains import NUM_PREDICTED_ANGLES, SC_DATA
from structure_exceptions import NonStandardAminoAcidError, IncompleteStructureError

GLOBAL_PAD_CHAR = np.nan
NUM_PREDICTED_COORDS = 13


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
        if aa in AA_MAP.keys():
            one_hot[AA_MAP[aa]] = 1
        else:
            one_hot -= 1
        vector_array.append(one_hot)
    return np.asarray(vector_array)


def onehot_to_seq(oh):
    """ Given a vector of one-hot vectors, returns its corresponding AA sequence."""
    seq = ""
    for aa in oh:
        idx = aa.argmax()
        seq += AA_MAP_INV[idx]
    return seq


def check_standard_continuous(residue, prev_res_num):
    """ Asserts that the residue is standard and that the chain is continuous. """
    if not residue.isstdaa:
        raise NonStandardAminoAcidError("Found a non-std AA.")
    if residue.getResnum() != prev_res_num:
        raise IncompleteStructureError("Chain is missing residues.")
    return True


def compute_sidechain_dihedrals(atom_names, residue, prev_residue, next_res):
    """
    Computes all angles to predict for a given residue. If the residue is the first in the
    protein chain, a fictitious C atom is placed before the first N. This is used to compute a [
    C-1, N, CA, CB] dihedral angle. If it is not the first residue in the chain, the previous
    residue's C is used instead. Then, each group of 4 atoms in atom_names is used to generate a
    list of dihedral angles for this residue.
    """
    res_dihedrals = []
    if len(atom_names) > 0:
        # The first residue has no previous residue, so use the next residue to calculate the (N+1)-C-CA-CB dihedral
        if prev_residue is None:
            atoms = [residue.select("name " + an) for an in atom_names]
            try:
                cb_dihedral = get_dihedral(next_res.select("name N").getCoords()[0],
                                           residue.select("name C").getCoords()[0],
                                           residue.select("name CA").getCoords()[0],
                                           residue.select("name CB").getCoords()[0],
                                           radian=True)
            except AttributeError:
                cb_dihedral = GLOBAL_PAD_CHAR
            res_dihedrals.append(cb_dihedral)
        # If there is a previous residue, use its C to calculate the C-N-CA-CB dihedral
        elif prev_residue is not None:
            atoms = [prev_residue.select("name C")] + [residue.select("name " + an) for an in atom_names]

        for n in range(len(atoms) - 3):
            dihe_atoms = atoms[n:n + 4]
            res_dihedrals.append(compute_single_dihedral(dihe_atoms))

    resname = residue.getResname()
    if resname not in ["LEU", "ILE", "VAL", "THR"]:
        return res_dihedrals + (NUM_PREDICTED_ANGLES - 6 - len(res_dihedrals)) * [GLOBAL_PAD_CHAR]
    # Extra angles to predict that are not included in SC_DATA[RES]["predicted"]
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
    res_dihedrals.append(compute_single_dihedral(atom_selections))
    assert len(res_dihedrals) == 4 and resname in ["ILE","LEU"] or len(res_dihedrals) == 3 and resname in ["VAL",
                                                                                                           "THR"], \
        "Angle position in array must match what is assumed in Sidechains:extend_any_sidechain."

    return res_dihedrals + (NUM_PREDICTED_ANGLES - 6 - len(res_dihedrals)) * [GLOBAL_PAD_CHAR]


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
        calculated_dihedrals = compute_sidechain_dihedrals(atom_names, res, prev_res, next_res)
        dihedrals.append(calculated_dihedrals)
        prev_res = res

    dihedrals_np = np.asarray(dihedrals)
    return dihedrals_np, sequence


def get_atom_coords_by_names(residue, atom_names):
    """
    Given a ProDy Residue and a list of atom names, this attempts to select and return all the atoms. If atoms are
    not present, it substitutes the pad character in lieu of their coordinates.
    """
    coords = []
    pad_coord = np.asarray([GLOBAL_PAD_CHAR]*3)
    for an in atom_names:
        a = residue.select(f"name {an}")
        if a:
            coords.append(a.getCoords()[0])
        else:
            coords.append(pad_coord)
    return coords


def get_seq_and_masked_coords_and_angles(chain):
    """
    Given a ProDy Chain object (from a Hierarchical View), return a tuple (angles, coords, sequence).
    Returns None if the PDB should be ignored due to weird artifacts. Also measures the
    bond angles along the peptide backbone, since they account for significant variation.
    i.e. [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2, chi3, chi4, chi5], [...] ...]
    """
    coords = []
    dihedrals = []
    observed_sequence = ""
    if chain.nonstdaa:
        raise NonStandardAminoAcidError
    chain = chain.select("protein and not hetero").copy()

    all_residues = list(chain.iterResidues())
    prev_res = None
    next_res = all_residues[1]
    for res_id, res in enumerate(all_residues):
        # Measure basic angles
        bb_angles = measure_phi_psi_omega(res)
        bond_angles = measure_bond_angles(res, res_id, all_residues)

        # Determine list of sidechain atoms over which to measure dihedrals
        if res.getResname() is "GLY":
            atom_names = []
        elif res.getResname() in SC_DATA.keys():
            atom_names = ["N", "CA"] + SC_DATA[res.getResname()]["predicted"]
        else:
            print(f"NSAA {chain}")
            raise NonStandardAminoAcidError
        all_res_angles = bb_angles + bond_angles + compute_sidechain_dihedrals(atom_names, res, prev_res, next_res)

        # Measure coordinates
        bbcoords = get_atom_coords_by_names(res, ["N", "CA", "C"])
        sccoords = get_atom_coords_by_names(res, set(atom_names) - {"N", "CA", "C", "H"})
        coord_padding = np.zeros((NUM_PREDICTED_COORDS - len(bbcoords) - len(sccoords), 3))
        coord_padding[:] = GLOBAL_PAD_CHAR
        rescoords = np.concatenate((np.stack(bbcoords + sccoords), coord_padding))

        coords.append(rescoords)
        dihedrals.append(all_res_angles)
        prev_res = res
        observed_sequence += res.getSequence()[0]

    dihedrals_np = np.asarray(dihedrals)
    coords_np = np.concatenate(coords)
    assert coords_np.shape[0] == len(observed_sequence) * NUM_PREDICTED_COORDS,   f"Coords shape {coords_np.shape} " \
        f"does not match len(seq)*13 = {len(observed_sequence) * NUM_PREDICTED_COORDS}"
    return dihedrals_np, coords_np, observed_sequence


def additional_checks(matrix):
    """ Returns true if a matrix does not contain NaNs, infs, or all 0s."""
    return not np.any(np.isinf(matrix)) and np.any(matrix)


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
    """
    Given 2 residues, returns the ncac, cacn, and cnca bond angles between them. If any atoms are not present,
    the corresponding angles are set to the GLOBAL_PAD_CHAR. If next_res is None, then only NCAC is measured.
    """
    # First residue angles
    n1, ca1, c1 = tuple(res.select(f"name {a}") for a in ["N", "CA", "C"])
    if n1 and ca1 and c1:
        ncac = pr.calcAngle(n1, ca1, c1, radian=True)[0]
    else:
        ncac = GLOBAL_PAD_CHAR

    # Second residue angles
    if next_res is None:
        return ncac, GLOBAL_PAD_CHAR, GLOBAL_PAD_CHAR
    n2, ca2 = (next_res.select(f"name {a}") for a in ["N", "CA"])
    if ca1 and c1 and n2:
        cacn = pr.calcAngle(ca1, c1, n2, radian=True)[0]
    else:
        cacn = GLOBAL_PAD_CHAR
    if c1 and n2 and ca2:
        cnca = pr.calcAngle(c1, n2, ca2, radian=True)[0]
    else:
        cnca = GLOBAL_PAD_CHAR
    return ncac, cacn, cnca


def measure_bond_angles(residue, res_idx, all_res):
    """ Given a residue, measure the ncac, cacn, and cnca bond angles. """
    if res_idx == len(all_res) - 1:
        next_res = None
    else:
        next_res = all_res[res_idx + 1]
    return list(get_bond_angles(residue, next_res))


def measure_phi_psi_omega(residue):
    """
    Returns phi, psi, omega for a residue, replacing out-of-bounds angles with
    GLOBAL_PAD_CHAR.
    """
    try:
        phi = pr.calcPhi(residue, radian=True, dist=None)
    except ValueError:
        phi = GLOBAL_PAD_CHAR
    try:
        psi = pr.calcPsi(residue, radian=True, dist=None)
    except ValueError:
        psi = GLOBAL_PAD_CHAR
    try:
        omega = pr.calcOmega(residue, radian=True, dist=None)
    except ValueError:
        omega = GLOBAL_PAD_CHAR
    return [phi, psi, omega]


def compute_single_dihedral(atoms):
    """
    Given an iterable of 4 Atoms, uses Prody to calculate the dihedral angle between them in
    radians.
    """
    if None in atoms:
        return GLOBAL_PAD_CHAR
    else:
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
AA_MAP_INV = {v: k for k, v in AA_MAP.items()}