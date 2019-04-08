"""" This script will take a trained model and a target dataset to predict on and make predictions that can be
     viewed as PDB files. """

import argparse
import os
import sys

sys.path.append("/home/jok120/sml/proj/attention-is-all-you-need-pytorch/")

import torch
from tqdm import tqdm
from prody import *
import numpy as np

import transformer.Models
import torch.utils.data
from dataset import ProteinDataset, paired_collate_fn
from transformer.Structure import generate_coords_with_tuples
from train import cal_loss
from losses import inverse_trig_transform, copy_padding_from_gold
from transformer.Sidechains import SC_DATA


def load_model(args):
    """ Given user-supplied arguments such as a model checkpoint, loads and returns the specified transformer model.
        If the data to predict is not specified, the original file used during training will be re-used. """
    device = torch.device('cpu')
    chkpt = torch.load(args.model_chkpt, map_location=device)
    model_args = chkpt['settings']
    model_state = chkpt['model']
    if args.data is None:
        args.data = model_args.data

    the_model = transformer.Models.Transformer(model_args.max_token_seq_len,
                                               d_k=model_args.d_k,
                                               d_v=model_args.d_v,
                                               d_model=model_args.d_model,
                                               d_inner=model_args.d_inner_hid,
                                               n_layers=model_args.n_layers,
                                               n_head=model_args.n_head,
                                               dropout=model_args.dropout)
    the_model.load_state_dict(model_state)
    return args, the_model


def get_data_loader(data_dict, dataset, n):
    """ Given a complete dataset as a python dictionary file and one of {train/test/val/all} to make predictions from,
        this function selects n items at random from that dataset to predict. It then returns a DataLoader for those
        items, along with a list of ids.
        """
    to_predict = np.random.choice(data_dict[dataset]["ids"], n)  # ["2NLP_D", "3ASK_Q", "1SZA_C"]
    ids = []
    seqs = []
    angs = []
    for i, prot in enumerate(data_dict[dataset]["ids"]):
        if prot.upper() in to_predict:
            seqs.append(data_dict[dataset]["seq"][i])
            angs.append(data_dict[dataset]["ang"][i])
            ids.append(prot)
    assert len(seqs) == n and len(angs) == n

    data_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=seqs,
            angs=angs),
        num_workers=2,
        batch_size=1,
        collate_fn=paired_collate_fn,
        shuffle=False)
    return data_loader, ids


def make_predictions(the_model, data_loader):
    """ Given a loaded transformer model, and a dataloader of items to predict, this model returns a list of tuples.
        Each tuple is contains (backbone coord. matrix, sidechain coord. matrix, loss, nloss) for a single item."""
    coords_list = []
    losses = []

    with torch.no_grad():
        for batch in tqdm(data_loader, mininterval=2, desc=' - (Evaluation ', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = batch
            gold = tgt_seq[:]

            # forward
            pred = the_model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss = cal_loss(pred, gold, src_seq, torch.device('cpu'), combined=False)
            losses.append(loss)

            pred, gold = inverse_trig_transform(pred), inverse_trig_transform(gold)
            pred, gold = copy_padding_from_gold(pred, gold, torch.device('cpu'))

            all, bb, bb_tups, sc, aa_codes, atom_names = generate_coords_with_tuples(pred[0], pred.shape[1], src_seq[0],
                                                                                     torch.device('cpu'))
            coords_list.append((np.asarray(bb), bb_tups, sc, float(loss), aa_codes, atom_names))
    print("Avg Loss = {0:.2f}".format(np.mean(losses)))
    return coords_list


def fill_in_residue(resname, coords, bb_cords, atom_names):
    """ Given an amino acid that is partially predicted (only the atoms in ATOM_NAMES are predicted),
        this function returns a list of coords that represents the complete amino acid structure."""
    all_res_atoms = set(SC_DATA[resname]["all_atoms"][4:])  # ignores N CA C O
    pred_res_atoms = set(SC_DATA[resname]["pred_atoms"])
    atoms_not_predicted = all_res_atoms - pred_res_atoms

    completed_atoms = []

    # TODO Add all sidechains here
    if resname == "ALA":
        pass
    if resname == "ARG":  # NH2
        pass
    elif resname == "GLY":
        return []  # no sidechain

    return [np.zeros(3)]

def make_pdbs(id_coords_dict, outdir):
    """ Given a dictionary that maps PDB_ID -> pred_coordinate_tuple, this function parses the true PDB file and
        assigns coordinates to its atoms so that a PDB file can be generated."""
    os.makedirs(outdir, exist_ok=True)
    for pdb_chain, data in id_coords_dict.items():
        bb_coords, bb_tups, sc_tups, loss, aa_codes, atom_names = data
        pdb_id = pdb_chain.split('_')[0]
        chain_id = pdb_chain.split("_")[-1]
        print(pdb_id, chain_id)

        prot = parsePDB(pdb_id)

        # dealing with multiple coordinate sets
        if len(prot.getCoordsets()) > 1:
            for i in range(len(prot.getCoordsets())):
                if i == 0:
                    pass
                else:
                    prot.delCoordset(-1)

        # TODO Fill in oxygen position
        # Set backbone atoms
        backbone = prot.select('protein and chain ' + chain_id + ' and name N CA C')
        assert backbone.getCoords().shape == bb_coords.shape, "Backbone shape mismatch for " + pdb_chain
        backbone.setCoords(bb_coords)

        # Set sidechain atoms
        assert len(aa_codes) == len(sc_tups) and len(sc_tups) == len(
            atom_names), "Shape mismatch for coordinate tuples."
        predicted_sidechain_coords = []
        for res_code, res_coords, res_bb_coords, res_atom_names in zip(aa_codes, sc_tups, bb_tups, atom_names):
            predicted_sidechain_coords.extend(fill_in_residue(res_code, res_coords, res_bb_coords, res_atom_names))

        sidechain = prot.select('protein and chain ' + chain_id + ' and sidechain')
        sidechain.setCoords(predicted_sidechain_coords)

        writePDB(os.path.join(outdir, pdb_chain + '_l{0:.2f}.pdb'.format(loss)), sidechain + backbone)


if __name__ == "__main__":
    np.random.seed(11)
    pathPDBFolder("/home/jok120/build/pdb/")
    parser = argparse.ArgumentParser(description="Loads a model and makes predictions as PDBs.")
    parser.add_argument('model_chkpt', type=str,
                        help="Path to model checkpoint file.")
    parser.add_argument("outdir", type=str,
                        help="Output directory to save predictions.")
    parser.add_argument("-data", type=str, required=False,
                        help="Path to data dictionary to predict. Defaults to test set from the data file that the" + \
                             " model was originally associated with.")
    parser.add_argument("-dataset", type=str, choices=["train", "valid", "test", "all"], default="test",
                        help="Which dataset within the data file to predict on.")
    parser.add_argument("-n", type=int, default=5, required=False,
                        help="How many items to randomly predict from dataset.")

    args = parser.parse_args()

    # Load model
    args, the_model = load_model(args)

    # Acquire seqs and angles to predict / compare from
    data_loader, ids = get_data_loader(torch.load(args.data), args.dataset, n=args.n)

    # Make predictions as coordinates
    coords_list = make_predictions(the_model, data_loader)
    id_coords_dict = {k: v for k, v in zip(ids, coords_list)}

    # Make PDB files from coords
    make_pdbs(id_coords_dict, args.outdir)
