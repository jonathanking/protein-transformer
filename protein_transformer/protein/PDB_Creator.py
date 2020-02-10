import numpy as np
import pymol
import torch
import wandb
from prody import calcTransformation

import os

from protein_transformer.protein.Sequence import ONE_TO_THREE_LETTER_MAP
import protein_transformer
from protein_transformer.losses import inverse_trig_transform
from protein_transformer.losses import angles_to_coords
from protein_transformer.protein.SidechainBuildInfo import SC_BUILD_INFO
from protein_transformer.protein.Structure import NUM_PREDICTED_COORDS


class PDB_Creator(object):
    """
    A class for creating PDB files given an atom mapping and coordinate set.
    The general idea is that if any model is capable of predicting a set of
    coordinates and mapping between those coordinates and residue/atom names,
    then this object can be use to transform that output into a PDB file.

    The Python format string was taken from http://cupnet.net/pdb-format/.
    """

    def __init__(self, coords, seq=None, mapping=None, atoms_per_res=NUM_PREDICTED_COORDS):
        """
        Input:
            coords: (L x N) x 3 where L is the protein sequence len and N
                    is the number of atoms/residue in the coordinate set
                    (atoms_per_res).
            mapping: length L list of list of atom names per residue,
                     i.e. (RES -> [ATOM_NAMES_FOR_RES])
        Output:
            saves PDB file to disk
        """

        self.coords = coords
        if seq and not mapping:
            assert len(seq) == coords.shape[0] / atoms_per_res, "The sequence length must match the coordinate length" \
                                                                " and contain 1 letter AA codes." + \
                                                                str(coords.shape[0]) + " " + str(len(seq))
            self.seq = seq
            self.mapping = self._make_mapping_from_seq()
        elif not seq and not mapping:
            raise Exception("Please provide a seq or a mapping.")
        elif mapping and not seq:
            self.mapping = mapping
            self.seq = self._get_seq_from_mapping()
        assert type(self.mapping[0][0]) == str and len(self.mapping[0][0]) == 1, "1 letter AA codes must be used in the mapping."
        self.atoms_per_res = atoms_per_res
        self.format_str = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          " \
                          "{:>2s}{:2s}"
        self.atom_nbr = 1
        self.res_nbr = 1
        self.defaults = {"alt_loc": "",
                         "chain_id": "",
                         "insertion_code": "",
                         "occupancy": 1,
                         "temp_factor": 0,
                         "element_sym": "",
                         "charge": ""}
        assert self.coords.shape[0] % self.atoms_per_res == 0, f"Coords is not divisible by {atoms_per_res}. " \
                                                               f"{self.coords.shape}"
        self.peptide_bond_full = np.asarray([[0.519, -2.968, 1.340],  # CA
                                             [2.029, -2.951, 1.374],  # C
                                             [2.654, -2.667, 2.392],  # O
                                             [2.682, -3.244, 0.300]])  # next-N
        self.peptide_bond_mobile = np.asarray([[0.519, -2.968, 1.340],  # CA
                                               [2.029, -2.951, 1.374],  # C
                                               [2.682, -3.244, 0.300]])  # next-N

    def _get_oxy_coords(self, ca, c, n):
        """
        Given atomic coordinates for an alpha carbon, carbonyl carbon, and the
        following nitrogen, this function aligns a peptide bond to those
        atoms and returns the coordinates of the corresponding oxygen.
        """
        target_coords = np.array([ca, c, n])
        t = calcTransformation(self.peptide_bond_mobile, target_coords)
        aligned_peptide_bond = t.apply(self.peptide_bond_full)
        return aligned_peptide_bond[2]

    def _coord_generator(self):
        """
        A generator that yields ATOMS_PER_RES atoms at a time from self.coords.
        """
        coord_idx = 0
        while coord_idx < self.coords.shape[0]:
            if coord_idx + self.atoms_per_res + 1 < self.coords.shape[0]:
                next_n = self.coords[coord_idx + self.atoms_per_res + 1]
            else:
                # TODO: Fix oxygen placement for final residue
                next_n = self.coords[-1] + np.array([1.2, 0, 0])
            yield self.coords[coord_idx:coord_idx + self.atoms_per_res], next_n
            coord_idx += self.atoms_per_res

    def _get_line_for_atom(self, res_name, atom_name, atom_coords, missing=False):
        """
        Returns the 'ATOM...' line in PDB format for the specified atom. If
        missing, this function should have special, but not yet determined,
        behavior.
        """
        if missing:
            occupancy = 0
        else:
            occupancy = self.defaults["occupancy"]
        return self.format_str.format("ATOM",
                                      self.atom_nbr,

                                      atom_name,
                                      self.defaults["alt_loc"],
                                      ONE_TO_THREE_LETTER_MAP[res_name],

                                      self.defaults["chain_id"],
                                      self.res_nbr,
                                      self.defaults["insertion_code"],

                                      atom_coords[0],
                                      atom_coords[1],
                                      atom_coords[2],
                                      occupancy,
                                      self.defaults["temp_factor"],

                                      atom_name[0],
                                      self.defaults["charge"])

    def _get_lines_for_residue(self, res_name, atom_names, coords, next_n):
        """
        Returns PDB-formated lines for all atoms in this residue. Calls
        get_line_for_atom.
        """
        residue_lines = []
        for atom_name, atom_coord in zip(atom_names, coords):
            # TODO: what to do in the PDB file if atom is missing?
            if atom_name is "PAD" or np.isnan(atom_coord).sum() > 0 or atom_coord.sum() == 0:
                continue
            # if np.isnan(atom_coord).sum() > 0:
            #     residue_lines.append(self.get_line_for_atom(res_name, atom_name, atom_coord,
            #     missing=True))
            #     self.atom_nbr += 1
            #     continue
            residue_lines.append(self._get_line_for_atom(res_name, atom_name, atom_coord))
            self.atom_nbr += 1
        if len(residue_lines) > 0:
            try:
                oxy_coords = self._get_oxy_coords(coords[1], coords[2], next_n)
                residue_lines.append(self._get_line_for_atom(res_name, "O", oxy_coords))
                self.atom_nbr += 1
            except ValueError:
                pass
        return residue_lines

    def _get_lines_for_protein(self):
        """
        Returns PDB-formated lines for all residues in this protein. Calls
        get_lines_for_residue.
        """
        self.lines = []
        self.res_nbr = 1
        self.atom_nbr = 1
        mapping_coords = zip(self.mapping, self._coord_generator())
        prev_n = torch.tensor([0, 0, -1])
        for (res_name, atom_names), (res_coords, next_n) in mapping_coords:
            self.lines.extend(self._get_lines_for_residue(res_name, atom_names, res_coords, next_n))
            prev_n = res_coords[0]
            self.res_nbr += 1
        return self.lines

    @staticmethod
    def _make_header(title):
        """
        Returns the PDB header.
        """
        return f"REMARK  {title}"

    @staticmethod
    def _make_footer():
        """
        Returns the PDB footer.
        """
        return "TER\nEND          \n"

    def _make_mapping_from_seq(self):
        """
        Given a protein sequence, this returns a mapping that assumes coords
        are generated in groups of 13, i.e. the output is L x 13 x 3.
        """
        mapping = []
        for residue in self.seq:
            mapping.append((residue, ATOM_MAP_13[residue]))
        return mapping

    def save_pdb(self, path, title="test"):
        """
        Given a file path and title, this function generates the PDB lines,
        then writes them to a file.
        """
        self._get_lines_for_protein()
        self.lines = [self._make_header(title)] + self.lines + [self._make_footer()]
        with open(path, "w") as outfile:
            outfile.write("\n".join(self.lines))

    def save_gltf(self, path, title="test", create_pdb=False):
        """
        This function first creates a PDB file, then converts it to a GLTF
        (3D Object) file. Used for visualizign with Weights and Biases. """
        assert ".gltf" in path, "requested filepath must end with '.gtlf'."
        if create_pdb:
            self.save_pdb(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.load(path.replace(".gltf", ".pdb"), title)
        pymol.cmd.color("oxygen", title)
        pymol.cmd.save(path, quiet=True)
        pymol.cmd.delete("all")

    def save_gltfs(self, path1, path2, gltf_out_path, make_pse=False, pse_out_path=""):
        """
        This function first creates a PDB file, then converts it to a GLTF
        (3D Object) file. Used for visualizign with Weights and Biases. """
        assert ".pdb" in path1, "requested filepaths must end with '.pdb'."
        pymol.cmd.load(path1, "true")
        pymol.cmd.load(path2, "pred")
        pymol.cmd.color("marine", "true")
        pymol.cmd.color("oxygen", "pred")
        rmsd, _, _, _, _, _, _ = pymol.cmd.align("true", "pred", quiet=True)
        pymol.cmd.save(gltf_out_path, f"{wandb.run.step:05}.gltf")

        # Align and save PSE
        if make_pse:
            pymol.cmd.save(pse_out_path, quiet=True)


        pymol.cmd.delete("all")


    def _get_seq_from_mapping(self):
        """
        Returns the protein sequence in 1-letter AA codes.
        """
        return "".join([m[0] for m in self.mapping])


ATOM_MAP_13 = {}
for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
    ATOM_MAP_13[one_letter] = ["N", "CA", "C"] + list(SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_13[one_letter].extend(["PAD"] * (13 - len(ATOM_MAP_13[one_letter])))

def generate_pdbs_from_debug_dataset():
    import torch

    data = torch.load("debug_struct.pt")

    seq, seq_gap = data["seq"]
    ang, ang_gap = data["ang"]
    crd, crd_gap = data["crd"]

    creator_from_crd = PDB_Creator(crd, seq=seq)
    creator_from_crd.save_pdb("from_crd.pdb")

    creator_from_crd_gap = PDB_Creator(crd_gap, seq=seq_gap)
    creator_from_crd_gap.save_pdb("from_crd_gap.pdb")

    crd_from_ang = get_coordinates_from_numpy_data(seq, ang)
    creator_from_ang = PDB_Creator(crd_from_ang, seq=seq)
    creator_from_ang.save_pdb("from_ang.pdb")

    crd_from_ang_gap = get_coordinates_from_numpy_data(seq_gap, ang_gap)
    creator_from_ang_gap = PDB_Creator(crd_from_ang_gap, seq=seq_gap)
    creator_from_ang_gap.save_pdb("from_ang_gap.pdb")


def get_coordinates_from_numpy_data(seq, ang_sincos):
    # Add batch dimension, make copy
    ang_sincos_new = ang_sincos[np.newaxis, :]

    # Compute angles in radians from sin/cos representaion
    ang_rad = \
    inverse_trig_transform(torch.tensor(ang_sincos_new, dtype=torch.float))[0]

    # Remove nans
    ang_rad[torch.isnan(ang_rad)] = 0

    if torch.isnan(ang_rad).any():
        print("Nan in ang_rad.")

    seq_as_ints = protein_transformer.dataset.VOCAB.str2ints(seq,
                                                             add_sos_eos=False)
    seq_as_ints = torch.tensor(seq_as_ints, dtype=torch.long)

    coords = angles_to_coords(ang_rad, seq_as_ints, remove_batch_padding=False)
    return coords.numpy()


def make_debug_structure_dataset():
    """ Creates a pytorch dictionary that contains one example of a structure
        without a gap, and another with a gap. Prints their ProteinNet IDs.
    """
    import torch

    data = torch.load("../../data/proteinnet/casp12_191101_100.pt")


    nogap = 3
    gap = 10#5
    seq, ang, crd = data["train"]["seq"][nogap], data["train"]["ang"][nogap], \
                    data["train"]["crd"][nogap]
    seq_gap, ang_gap, crd_gap = data["train"]["seq"][gap], data["train"]["ang"][
        gap], \
                                data["train"]["crd"][gap]
    print(data["train"]["ids"][nogap])
    print(data["train"]["ids"][gap])

    d = {"seq": (seq, seq_gap), "ang": (ang, ang_gap), "crd": (crd, crd_gap)}
    torch.save(d, "debug_struct.pt")


if __name__ == "__main__":
    # TODO don't reimplement dataloader?
    # TODO do several predictions, add PDB name
    # TODO add ability to predict given model checkpoint
    # TODO CUDA isn't playing nice

    make_debug_structure_dataset()
    generate_pdbs_from_debug_dataset()

