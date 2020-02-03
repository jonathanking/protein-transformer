import torch
import numpy as np

from protein_transformer.dataset import VOCAB, NUM_PREDICTED_COORDS
from protein_transformer.protein import Sidechains
from protein_transformer.protein.Sidechains import SC_DATA, BONDLENS
from protein_transformer.protein.Structure import nerf

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
        self.seq = seq
        self.ang = ang
        self.device = device
        self.coords = []
        self.prev_ang = None
        self.prev_bb = None

    def iter_residues(self):
        for resname, angles in zip(self.seq, self.ang):
            yield ResidueBuilder(resname, angles, self.prev_bb, self.prev_ang) 

    def build(self):
        for residue in self.iter_residues():
            residue.build()
            self.coords += residue.coords
            self.prev_ang = residue.ang
            self.prev_bb = residue.bb

        return self.coords



class ResidueBuilder(object):

    def __init__(self, name, angles, prev_bb, prev_ang):
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
        assert len(name) == 1 and type(name) == torch.Tensor, "Expected integer AA code."
        self.name = name
        self.ang = angles
        self.prev_bb = prev_bb
        self.prev_ang = prev_ang

        self.bb = []
        self.sc = []
        self.coords = []
        self.coordinate_padding = torch.zeros(3)

    def build(self):
        self.build_bb()
        self.build_sc()
        self.stack_coords()

    def build_bb(self):
        """ Builds backbone for residue. """
        if self.prev_ang is None and self.prev_bb is None:
            self.bb = self.init_bb()
        else:
            self.bb = []
            for j in range(3):
                if j == 0:
                    # Placing N
                    t = self.prev_ang[4]         # thetas["ca-c-n"]
                    b = BONDLENS["c-n"]
                    dihedral = self.prev_ang[1]  # psi of previous residue
                elif j == 1:
                    # Placing Ca
                    t = self.prev_ang[5]         # thetas["c-n-ca"]
                    b = BONDLENS["n-ca"]
                    dihedral = self.prev_ang[2]  # omega of previous residue
                else:
                    # Placing C
                    t = self.ang[3]              # thetas["n-ca-c"]
                    b = BONDLENS["ca-c"]
                    dihedral = self.ang[0]       # phi of current residue

                next_pt = nerf(self.prev_bb[-3], self.prev_bb[-2], self.prev_bb[-1], b, t, dihedral)
                self.bb.append(next_pt)

        return self.bb

    def init_bb(self):
        """ Initialize the first 3 points of the protein's backbone. Placed in an arbitrary plane (x = .001). """
        a1 = torch.tensor([0.001, 0, 0], device=self.device)
        a2 = a1 + torch.tensor([BONDLENS["n-ca"], 0, 0], device=self.device)
        a3x = torch.cos(np.pi - self.ang[0, 3]) * BONDLENS["ca-c"]
        a3y = torch.sin(np.pi - self.ang[0, 3]) * BONDLENS['ca-c']
        a3 = a2 + torch.tensor([a3x, a3y, 0], device=self.device)
        return [a1, a2, a3]

    def build_sc(self):
        assert len(self.bb) > 0, "Backbone must be built first."
        pts = list(self.prev_bb[-1] + self.bb[:-1]) # Start with C, N, CA
        bond_lens = Sidechains.get_bond_len_iter(self.name)
        angles = Sidechains.get_angle_iter(self.name, predicted_angles=self.ang)
        torsions = Sidechains.get_torsion_iter(self.name, predicted_angles=self.ang)
        last_three_pts_slice = Sidechains.get_3_pts_indices_iter(self.name)
        for (start, end), bond_len, angle, torsion in zip(last_three_pts_slice, bond_lens, angles, torsions):
            a, b, c = pts[start:end]
            new_pt = nerf(a, b, c, bond_len, angle, torsion)
            pts.append(new_pt)

        self.sc = pts

    def stack_coords(self):
        self.coords = self.bb + self.sc + (NUM_PREDICTED_COORDS - \
            len(self.bb) - len(self.sc)) * [self.coordinate_padding]
