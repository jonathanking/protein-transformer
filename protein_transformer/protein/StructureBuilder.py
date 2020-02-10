import numpy
import numpy as np
import torch

from protein_transformer.protein.Sequence import VOCAB
from protein_transformer.protein.SidechainBuildInfo import SC_BUILD_INFO, \
    BB_BUILD_INFO
from protein_transformer.protein.Structure import nerf, NUM_PREDICTED_COORDS, \
    SC_ANGLE_START_POS


class StructureBuilder(object):
    """ 
    Given angles and protein sequence, reconstructs a single protein's structure.
    """
    def __init__(self, seq, ang, device=torch.device("cpu")):
        """
        Initialize a StructureBuilder for a single protein.

        Parameters
        ----------
        seq : Tensor
            An integer tensor (L) (without padding) that represents the protein's amino acid sequence.
        ang : Tensor
            An angle tensor (L X NUM_PREDICTED_ANGLES) that contain's all of the protein's interior angles.
        device : device
            The device on which to build the structure.
        """
        if type(seq) == str:
            seq = torch.tensor([VOCAB._char2int[s] for s in seq])
        self.seq = seq
        self.ang = ang
        self.device = device
        self.coords = []
        self.prev_ang = None
        self.prev_bb = None
        self.next_bb = None

        self.pdb_creator = None
        self.seq_as_str = None

    def iter_residues(self, start=0):
        for resname, angles in zip(self.seq[start:], self.ang[start:]):
            yield ResidueBuilder(resname, angles, self.prev_bb, self.prev_ang)

    def build(self):
        """
        Construct all of the atoms for a residue. Special care must be taken
        for the first residue in the sequence in order to place its CB, if
        present.
        """
        # Initialize the first and second residues' backbones
        residue_iterator = self.iter_residues()
        first_res = next(residue_iterator)
        first_res.build_bb()
        self.prev_bb, self.prev_ang = first_res.bb, first_res.ang
        second_res = next(residue_iterator)
        second_res.build()
        self.prev_bb, self.prev_ang = second_res.bb, second_res.ang

        # Use the second residue's bb to build the first's CB
        first_res.next_bb = second_res.bb
        first_res.build_sc()

        # Combine the coordinates and build the rest of the protein
        self.coords = first_res.stack_coords() + second_res.stack_coords()

        for residue in self.iter_residues(start=2):
            residue.build()
            self.coords += residue.coords
            self.prev_ang = residue.ang
            self.prev_bb = residue.bb

        self.coords = torch.stack(self.coords)

        return self.coords

    def get_seq_as_str(self):
        if not self.seq_as_str:
            self.seq_as_str = VOCAB.ints2str(map(int, self.seq))
        return self.seq_as_str

    def to_pdb(self, path, title="pred"):
        if len(self.coords) == 0:
            self.build()

        if not self.pdb_creator:
            from protein_transformer.protein.PDB_Creator import PDB_Creator
            self.pdb_creator = PDB_Creator(self.coords.numpy(), self.get_seq_as_str())

        self.pdb_creator.save_pdb(path, title)



class ResidueBuilder(object):

    def __init__(self, name, angles, prev_bb, prev_ang, device=torch.device("cpu")):
        """Initialize a residue builder. If prev_{bb, ang} are None, then this
        is the first residue.

        Parameters
        ----------
        name : Tensor
            The integer amino acid code for this residue.
        angles : Tensor
            Angle tensor containing necessary angles to define this residue.
        prev_bb : Tensor, None
            Coordinate tensor (3 x 3) of previous residue, upon which this residue is extending.
        prev_ang : Tensor, None
            Angle tensor (1 X NUM_PREDICTED_ANGLES) of previous reside, upon which this residue is extending.
        """
        assert type(name) == torch.Tensor, "Expected integer AA code." + str(name.shape) + str(type(name))
        if type(angles) == numpy.ndarray:
            angles = torch.tensor(angles, dtype=torch.float32)
        self.name = name
        self.ang = angles.squeeze()
        self.prev_bb = prev_bb
        self.prev_ang = prev_ang
        self.device = device
        self.next_bb = None

        self.bb = []
        self.sc = []
        self.coords = []
        self.coordinate_padding = torch.zeros(3)

    def build(self):
        self.build_bb()
        self.build_sc()
        return self.stack_coords()

    def build_bb(self):
        """ Builds backbone for residue. """
        if self.prev_ang is None and self.prev_bb is None:
            self.bb = self.init_bb()
        else:
            pts = [self.prev_bb[-3], self.prev_bb[-2], self.prev_bb[-1]]
            for j in range(3):
                if j == 0:
                    # Placing N
                    t = self.prev_ang[4]         # thetas["ca-c-n"]
                    b = BB_BUILD_INFO["BONDLENS"]["c-n"]
                    dihedral = self.prev_ang[1]  # psi of previous residue
                elif j == 1:
                    # Placing Ca
                    t = self.prev_ang[5]         # thetas["c-n-ca"]
                    b = BB_BUILD_INFO["BONDLENS"]["n-ca"]
                    dihedral = self.prev_ang[2]  # omega of previous residue
                else:
                    # Placing C
                    t = self.ang[3]              # thetas["n-ca-c"]
                    b = BB_BUILD_INFO["BONDLENS"]["ca-c"]
                    dihedral = self.ang[0]       # phi of current residue

                next_pt = nerf(pts[-3], pts[-2], pts[-1], b, t, dihedral)
                pts.append(next_pt)
            self.bb = pts[3:]

        return self.bb

    def init_bb(self):
        """ Initialize the first 3 points of the protein's backbone. Placed in an arbitrary plane (z = .001). """
        a1 = torch.tensor([0, 0, 0.001], device=self.device)
        a2 = a1 + torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"], 0, 0], device=self.device)
        a3x = torch.cos(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]["ca-c"]
        a3y = torch.sin(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]['ca-c']
        a3 = a2 + torch.tensor([a3x, a3y, 0], device=self.device, dtype=torch.float32)
        return [a1, a2, a3]

    def build_sc(self):
        """
        Builds the sidechain atoms for this residue.

        Care is taken when placing the first sc atom (the beta-Carbon). This is
        because the dihedral angle that places this atom must be defined using
        a neighboring (previous or next) residue.
        """
        assert len(self.bb) > 0, "Backbone must be built first."
        self.pts = {"N": self.bb[0],
                    "CA": self.bb[1],
                    "C": self.bb[2]}
        if self.next_bb:
            self.pts["N+"] = self.next_bb[0]
        else:
            self.pts["C-"] = self.prev_bb[-1]

        last_torsion = None
        for i, (bond_len, angle, torsion, atom_names) in enumerate(get_residue_build_iter(self.name, SC_BUILD_INFO)):

            if self.next_bb and i == 0:
                a, b, c = self.pts["N+"], self.pts["C"], self.pts["CA"]
            elif i == 0:
                a, b, c = self.pts["C-"], self.pts["N"], self.pts["CA"]
            else:
                a, b, c = (self.pts[an] for an in atom_names[:-1])

            if type(torsion) is str and torsion == "p":
                torsion = self.ang[SC_ANGLE_START_POS + i]
            elif type(torsion) is str and torsion == "i" and last_torsion:
                torsion = last_torsion - np.pi
            new_pt = nerf(a, b, c, bond_len, angle, torsion)
            self.pts[atom_names[-1]] = new_pt
            self.sc.append(new_pt)
            last_torsion = torsion

        return self.sc

    def stack_coords(self):
        self.coords = self.bb + self.sc + (NUM_PREDICTED_COORDS - \
            len(self.bb) - len(self.sc)) * [self.coordinate_padding]
        return self.coords

    def __repr__(self):
        return f"ResidueBuilder({VOCAB.int2char(self.name)})"

def get_residue_build_iter(res, build_dictionary):
    """
    For a given residue integer code and a residue building data dictionary,
    this function returns an iterator that returns 4-tuples. Each tuple
    contains the necessary information to generate the next atom in that
    residue's sidechain. This includes the bond lengths, bond angles, and
    torsional angles.
    """
    r = build_dictionary[VOCAB.int2chars(int(res))]
    bvals = [torch.tensor(b, dtype=torch.float32) for b in r["bonds-vals"]]
    avals = [torch.tensor(a, dtype=torch.float32) for a in r["angles-vals"]]
    tvals = [torch.tensor(t, dtype=torch.float32) if t not in ["p", "i"] else t for t in r["torsion-vals"]]
    return iter(zip(bvals, avals, tvals, [t.split("-") for t in r["torsion-names"]]))

if __name__ == '__main__':
    a = get_residue_build_iter("ALA", SC_BUILD_INFO)
    b = get_residue_build_iter("ARG", SC_BUILD_INFO)
    c = get_residue_build_iter("TYR", SC_BUILD_INFO)
    for i in a:
        print(i)
    print("Arginine:")
    for i in b:
        print(f"\t{i}")
    print("Tyrosine:")
    for i in c:
        print(f"\t{i}")