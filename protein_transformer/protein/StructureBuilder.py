import torch

from protein_transformer.dataset import VOCAB, NUM_PREDICTED_COORDS
from protein_transformer.Sidechains import SC_DATA, BONDLENS

class StructureBuilder(object):
    """ 
    Given angles and protein sequence, reconstructs the protein's structure.
    """

    def __init__(self, seq, ang, device=torch.device("cpu")):
        self.seq = seq
        self.ang = ang
        self.device = device

        # TODO: what type is seq? What type is most convenient for building str?
        self.coords = []        

    def init_bb(self):
        # TODO: think about this, update. Ficticious residue?
        a1 = torch.tensor([0.001, 0, 0], device=self.device)
        a2 = a1 + torch.tensor([BONDLENS["n-ca"], 0, 0], device=self.device)
        a3x = torch.cos(np.pi - angles[0, 3]) * BONDLENS["ca-c"]
        a3y = torch.sin(np.pi - angles[0, 3]) * BONDLENS['ca-c']
        a3 = a2 + torch.tensor([a3x, a3y, 0], device=self.device)
        self.bb = [a1, a2, a3]

    def iter_residues(self):
        for resname, angles in zip(seq, ang):
            yield ResidueBuilder(resname, angles, self.prev_bb, self.prev_ang) 

    def build(self):
        self.prev_ang = None
        self.prev_bb = self.init_bb()

        for residue in self.iter_residues():
            residue.build()
            self.coords += residue.coords
            self.prev_ang = residue.ang
            self.prev_bb = residue.bb

        return self.coords



class ResidueBuilder(object):

    def __init__(self, name, angles prev_bb, prev_ang)
        assert len(name) == 3 and type(name) == str, "use 3 letter AA code, or change data structure"
        self.name = name
        self.ang = angles
        self.prev_bb = prev_bb
        self.prev_ang = prev_ang

        self.bb = []
        self.sc = []
        self.coords = []

    def build(self):
        self.build_bb()
        self.build_sc()
        self.stack_coords()

    def build_bb(self):
        """ Builds backbone for reside. """
        # TODO Assert programatically that angles are in the right order
        bb_pts = []
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
            bb_pts.append(next_pt)

        self.bb = bb_pts
        return bb_pts


    def build_sc(self):
        self.sc = Sidechains.build(self)

    def stack_coords(self):
        self.coords = self.bb + self.sc + (NUM_PREDICTED_COORDS - \
            len(self.bb) - len(self.sc)) * [torch.zeros(3, device=device)]
