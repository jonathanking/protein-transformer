import numpy as np
from prody import calcTransformation
import torch
from protein.Sidechains import SC_DATA, ONE_TO_THREE_LETTER_MAP, THREE_TO_ONE_LETTER_MAP


class PDB_Creator(object):
    """
    A class for creating PDB files given an atom mapping and coordinate set.
    The general idea is that if any model is capable of predicting a set of
    coordinates and mapping between those coordinates and residue/atom names,
    then this object can be use to transform that output into a PDB file.

    The Python format string was taken from http://cupnet.net/pdb-format/.
    """

    def __init__(self, coords, seq=None, mapping=None, atoms_per_res=13):
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
            assert len(seq) == coords.shape[0] / atoms_per_res, "The sequence length must match the coordinate length and contain 1 " \
                                                "letter AA codes." + str(coords.shape[0]) + " " + str(len(seq))
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
        iven atomic coordinates for an alpha carbon, carbonyl carbon, and the
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
            if atom_name is "PAD" or np.isnan(atom_coord).sum() > 0:
                continue
            # if np.isnan(atom_coord).sum() > 0:
            #     residue_lines.append(self.get_line_for_atom(res_name, atom_name, atom_coord,
            #     missing=True))
            #     self.atom_nbr += 1
            #     continue
            residue_lines.append(self._get_line_for_atom(res_name, atom_name, atom_coord))
            self.atom_nbr += 1
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
        print(f"PDB {title} written to {path}.")

    def _get_seq_from_mapping(self):
        """
        Returns the protein sequence in 1-letter AA codes.
        """
        return "".join([m[0] for m in self.mapping])


def load_model(chkpt_path):
    """
    Given a checkpoint path, loads and returns the specified transformer model.
    """
    chkpt = torch.load(chkpt_path)
    model_args = chkpt['settings']
    model_args.cuda = True
    model_state = chkpt['model_state_dict']
    model_args.postnorm = False
    print(model_args)

    the_model = models.transformer.Models.Transformer(model_args,
                                                      d_k=model_args.d_k,
                                                      d_v=model_args.d_v,
                                                      d_model=model_args.d_model,
                                                      d_inner=model_args.d_inner_hid,
                                                      n_layers=model_args.n_layers,
                                                      n_head=model_args.n_head,
                                                      dropout=model_args.dropout)
    the_model.load_state_dict(model_state)
    the_model.use_cuda = True
    return the_model

def get_data_loader(data_path, n=0, subset="test"):
    """
    Given a subset of a dataset as a python dictionary file to make
    predictions from, this function selects n items at random from that
    dataset to predict. It then returns a DataLoader for those items,
    along with a list of ids.
    """
    data = torch.load(data_path)
    data_subset = data[subset]

    if n is 0:
        train_loader = torch.utils.data.DataLoader(
            ProteinDataset(
                seqs=data_subset['seq'],
                crds=data_subset['crd'],
                angs=data_subset['ang'],
                ),
            num_workers=2,
            batch_size=1,
            collate_fn=paired_collate_fn,
            shuffle=False)
        return train_loader, data_subset["ids"]

    # We just want to predict a few examples
    to_predict = set([s.upper() for s in np.random.choice(data_subset["ids"], n)])  # ["2NLP_D", "3ASK_Q", "1SZA_C"]
    will_predict = []
    ids = []
    seqs = []
    angs = []
    crds = []
    for i, prot in enumerate(data_subset["ids"]):
        if prot.upper() in to_predict and prot.upper() not in will_predict:
            seqs.append(data_subset["seq"][i])
            angs.append(data_subset["ang"][i])
            crds.append(data_subset["crd"][i])
            ids.append(prot)
            will_predict.append(prot.upper())
    assert len(seqs) == n and len(angs) == n or (len(seqs) == len(angs) and len(seqs) < n)

    data_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=seqs,
            angs=angs,
            crds=crds),
        num_workers=2,
        batch_size=1,
        collate_fn=paired_collate_fn,
        shuffle=False)
    return data_loader, ids


def make_prediction(title, data_iter):
    src_seq, src_pos_enc, tgt_ang, tgt_pos_enc, tgt_crds, tgt_crds_enc = next(data_iter)
    pred = model.predict(src_seq, src_pos_enc)

    # Calculate loss
    d_loss, d_loss_normalized, r_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq[:, 1:], device,
                                                               return_rmsd=True)
    m_loss = mse_over_angles(pred, tgt_ang[:, 1:]).to('cpu')
    print(f"Losses:\n\tMSE = {m_loss.item():.4f}\n\tDRMSD = {d_loss.item():.4f}\n\t"
          f"ln-DRMSD = {d_loss_normalized.item():.5f}\n\tRMSD = {r_loss.item():.2f}")

    # Generate coords
    pred = inverse_trig_transform(pred).squeeze()
    src_seq = src_seq.squeeze()
    coords = generate_coords(pred, pred.shape[0], src_seq[:, 1:], device)

    # Generate coord, atom_name mapping
    one_letter_seq = onehot_to_seq(src_seq.squeeze().detach().numpy())
    cur_map = get_13atom_mapping(one_letter_seq)

    # Make PDB Creator objects
    pdb_pred = PDB_Creator(coords.squeeze(), cur_map)
    pdb_true = PDB_Creator(tgt_crds.squeeze(), cur_map)

    # Save PDB files
    pdb_pred.save_pdb(f"{title}_pred.pdb")
    pdb_true.save_pdb(f"{title}_true.pdb")

    # Align PDB files
    # TODO: fix alignment strategy
    # p = parsePDB(f"{title}_pred.pdb")
    # t = parsePDB(f"{title}_true.pdb")
    # tr = calcTransformation(p.getCoords()[:-1], t.getCoords())
    # p.setCoords(tr.apply(p.getCoords()))
    # writePDB(f"{title}_pred.pdb", p)

    print(f"Constructed PDB files for {title}.")


ATOM_MAP_13 = {}
for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
    ATOM_MAP_13[one_letter] = ["N", "CA", "C"] + list(SC_DATA[ONE_TO_THREE_LETTER_MAP[one_letter]]["predicted"])
    ATOM_MAP_13[one_letter].extend(["PAD"] * (13 - len(ATOM_MAP_13[one_letter])))

if __name__ == "__main__":
    # TODO Clean imports
    # TODO don't reimplement dataloader?
    # TODO do several predictions, add PDB name
    # TODO add ability to predict given model checkpoint
    # TODO CUDA isn't playing nice
    import sys
    import os
    os.chdir('/home/jok120/protein-transformer/')
    sys.path.append("/home/jok120/protein-transformer/scripts/utils")
    import torch.utils.data
    from dataset import ProteinDataset, paired_collate_fn
    from protein.Structure import generate_coords
    from losses import inverse_trig_transform, drmsd_loss_from_angles, mse_over_angles
    from scripts.utils.structure_utils import onehot_to_seq

    device = torch.device('cuda')

    model = load_model("data/checkpoints/casp12_30_ln_11_best.chkpt")
    data_loader, ids = get_data_loader('data/proteinnet/casp12_190809_30xsmall.pt')
    data_iter = iter(data_loader)
    make_prediction(9, data_iter)

