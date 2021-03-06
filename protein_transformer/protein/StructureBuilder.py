import numpy
import numpy as np
import torch

from protein_transformer.protein.Sequence import VOCAB
from protein_transformer.protein.SidechainBuildInfo import SC_BUILD_INFO, \
    BB_BUILD_INFO
from protein_transformer.protein.Structure import nerf, NUM_PREDICTED_COORDS, \
    SC_ANGLES_START_POS


class StructureBuilder(object):
    """
    Given angles and protein sequence, reconstructs a single protein's structure.

    The hydroxyl-oxygen of terminal residues is not placed because this would
    mean that the number of coordinates per residue would not be constant, or
    cause other complications (i.e. what if the last atom of a structure is not
    really a terminal atom because it's tail is masked out?). It is simpler to
    ignore this atom for now.
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

    def __len__(self):
        return len(self.seq)

    def iter_resname_angs(self, start=0):
        for resname, angles in zip(self.seq[start:], self.ang[start:]):
            yield resname, angles

    def build_first_two_residues(self):
        """ Constructs the first two residues of the protein. """
        resname_ang_iter = self.iter_resname_angs()
        first_resname, first_ang = next(resname_ang_iter)
        second_resname, second_ang = next(resname_ang_iter)
        first_res = ResidueBuilder(first_resname, first_ang, prev_res=None, next_res=None)
        second_res = ResidueBuilder(second_resname, second_ang, prev_res=first_res, next_res=None)

        # After building both backbones, use the second residue's N to build the first's CB
        first_res.build_bb()
        second_res.build()
        first_res.next_res = second_res
        first_res.build_sc()

        return first_res, second_res

    def build(self):
        """
        Construct all of the atoms for a residue. Special care must be taken
        for the first residue in the sequence in order to place its CB, if
        present.
        """
        # Build the first and second residues, a special case
        first, second = self.build_first_two_residues()

        # Combine the coordinates and build the rest of the protein
        self.coords = first.stack_coords() + second.stack_coords()

        # Build the rest of the structure
        prev_res = second
        for i, (resname, ang) in enumerate(self.iter_resname_angs(start=2)):
            res = ResidueBuilder(resname, ang, prev_res=prev_res, next_res=None)
            self.coords += res.build()
            prev_res = res

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

    def __init__(self, name, angles, prev_res, next_res, device=torch.device("cpu")):
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
        self.prev_res = prev_res
        self.next_res = next_res
        self.device = device

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
        if self.prev_res is None:
            self.bb = self.init_bb()
        else:
            pts = [self.prev_res.bb[0], self.prev_res.bb[1], self.prev_res.bb[2]]
            for j in range(4):
                if j == 0:
                    # Placing N
                    t = self.prev_res.ang[4]         # thetas["ca-c-n"]
                    b = BB_BUILD_INFO["BONDLENS"]["c-n"]
                    dihedral = self.prev_res.ang[1]  # psi of previous residue
                elif j == 1:
                    # Placing Ca
                    t = self.prev_res.ang[5]         # thetas["c-n-ca"]
                    b = BB_BUILD_INFO["BONDLENS"]["n-ca"]
                    dihedral = self.prev_res.ang[2]  # omega of previous residue
                elif j == 2:
                    # Placing C
                    t = self.ang[3]              # thetas["n-ca-c"]
                    b = BB_BUILD_INFO["BONDLENS"]["ca-c"]
                    dihedral = self.ang[0]       # phi of current residue
                else:
                    # Placing O
                    t = torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"])
                    b = BB_BUILD_INFO["BONDLENS"]["c-o"]
                    dihedral = self.ang[1] - np.pi      # opposite to psi of current residue

                next_pt = nerf(pts[-3], pts[-2], pts[-1], b, t, dihedral)
                pts.append(next_pt)
            self.bb = pts[3:]

        return self.bb

    def init_bb(self):
        """ Initialize the first 3 points of the protein's backbone. Placed in an arbitrary plane (z = .001). """
        n = torch.tensor([0, 0, 0.001], device=self.device)
        ca = n + torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"], 0, 0], device=self.device)
        cx = torch.cos(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]["ca-c"]
        cy = torch.sin(np.pi - self.ang[3]) * BB_BUILD_INFO["BONDLENS"]['ca-c']
        c = ca + torch.tensor([cx, cy, 0], device=self.device, dtype=torch.float32)
        o = nerf(n, ca, c, torch.tensor(BB_BUILD_INFO["BONDLENS"]["c-o"]),
                           torch.tensor(BB_BUILD_INFO["BONDANGS"]["ca-c-o"]),
                           self.ang[1] - np.pi) # opposite to current residue's psi
        return [n, ca, c, o]

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
        if self.next_res:
            self.pts["N+"] = self.next_res.bb[0]
        else:
            self.pts["C-"] = self.prev_res.bb[2]

        last_torsion = None
        for i, (bond_len, angle, torsion, atom_names) in enumerate(get_residue_build_iter(self.name, SC_BUILD_INFO)):
            # Select appropriate 3 points to build from
            if self.next_res and i == 0:
                a, b, c = self.pts["N+"], self.pts["C"], self.pts["CA"]
            elif i == 0:
                a, b, c = self.pts["C-"], self.pts["N"], self.pts["CA"]
            else:
                a, b, c = (self.pts[an] for an in atom_names[:-1])

            # Select appropriate torsion angle, or infer it if it's part of a planar configuration
            if type(torsion) is str and torsion == "p":
                torsion = self.ang[SC_ANGLES_START_POS + i]
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
        return f"ResidueBuilder({VOCAB.int2char(int(self.name))})"

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