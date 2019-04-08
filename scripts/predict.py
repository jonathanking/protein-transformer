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
import transformer.Structure as struct
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

    # TODO: make batch_level predictions?
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

            all, bb, sc, aa_codes, atom_names = struct.generate_coords(pred[0], pred.shape[1], src_seq[0],
                                                                       torch.device('cpu'),
                                                                       return_tuples=True)
            coords_list.append((np.asarray(bb), np.asarray(sc), float(loss), aa_codes, atom_names))
    print("Avg Loss = {0:.2f}".format(np.mean(losses)))
    return coords_list


def make_pdbs(id_coords_dict, outdir):
    """ Given a dictionary that maps PDB_ID -> pred_coordinate_tuple, this function parses the true PDB file and
        assigns coordinates to its atoms so that a PDB file can be generated."""
    os.makedirs(outdir, exist_ok=True)
    for key in id_coords_dict.keys():
        bb_coords, sc_coords, loss, aa_codes, atom_names = id_coords_dict[key]
        flat_atom_names = [an for res in atom_names for an in res]
        pdb_id = key.split('_')[0]
        chain_id = key.split("_")[-1]

        prot = parsePDB(pdb_id)
        print(pdb_id, chain_id)

        # dealing with multiple coordsets
        if len(prot.getCoordsets()) > 1:
            for i in range(len(prot.getCoordsets())):
                if i == 0:
                    pass
                else:
                    prot.delCoordset(-1)

        # Set backbone atoms
        backbone = prot.select('protein and chain ' + chain_id + ' and name N CA C')
        assert backbone.getCoords().shape == bb_coords.shape, "Backbone shape mismatch for " + key
        backbone.setCoords(bb_coords)

        # Set sidechain atoms
        # chains = [c for c in prot.select("protein and chain " + chain_id).getHierView()]
        # assert len(chains) == 1, "HV has more than one chain for " + key
        # residues = list(chains[0].iterResidues())
        # assert len(sc_coords) == len(residues)
        # for res_sc_coords, res in zip(sc_coords, residues):

        writePDB(os.path.join(outdir, key + '_l{0:.2f}.pdb'.format(loss)), predicted_sidechain + backbone)


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
