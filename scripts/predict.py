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
from train import drmsd_loss
from losses import inverse_trig_transform, copy_padding_from_gold
from transformer.Sidechains import SC_DATA


def load_model(args):
    """ Given user-supplied arguments such as a model checkpoint, loads and returns the specified transformer model.
        If the data to predict is not specified, the original file used during training will be re-used. """
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
    if dataset == "all":
        data_dict["all"] = {"ids": data_dict["train"]["ids"] + data_dict["test"]["ids"] + data_dict["valid"]["ids"],
                            "seq": data_dict["train"]["seq"] + data_dict["test"]["seq"] + data_dict["valid"]["seq"],
                            "ang": data_dict["train"]["ang"] + data_dict["test"]["ang"] + data_dict["valid"]["ang"]}
    to_predict = np.random.choice(data_dict[dataset]["ids"], n)  # ["2NLP_D", "3ASK_Q", "1SZA_C"]
    ids = []
    seqs = []
    angs = []
    for i, prot in enumerate(data_dict[dataset]["ids"]):
        if prot.upper() in to_predict:
            seqs.append(data_dict[dataset]["seq"][i])
            angs.append(data_dict[dataset]["ang"][i])
            ids.append(prot)
    assert len(seqs) == n and len(angs) == n or (len(seqs) == len(angs) and len(seqs) < n)

    data_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=seqs,
            angs=angs),
        num_workers=2,
        batch_size=1,
        collate_fn=paired_collate_fn,
        shuffle=False)
    return data_loader, ids


def make_predictions(the_model, data_loader, pdb_ids, build_true=False):
    """ Given a loaded transformer model, and a dataloader of items to predict, this model returns a list of tuples.
        Each tuple is contains (backbone coord. matrix, sidechain coord. matrix, loss, nloss) for a single item."""
    coords_list = []
    losses = []

    with torch.no_grad():
        for pdb_id, batch in zip(pdb_ids, tqdm(data_loader, mininterval=2, desc=' - (Evaluation ', leave=False)):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = batch
            gold = tgt_seq[:]

            # forward
            if args.reconstruct or build_true:
                pred = tgt_seq
            else:
                pred = the_model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss = drmsd_loss(pred, gold, src_seq, torch.device('cpu'))
            losses.append(loss)

            pred, gold = inverse_trig_transform(pred), inverse_trig_transform(gold)
            pred, gold = copy_padding_from_gold(pred, gold, torch.device('cpu'))

            all, bb, bb_tups, sc, aa_codes, atom_names = generate_coords_with_tuples(pred[0], pred.shape[1], src_seq[0],
                                                                                     torch.device('cpu'))
            coords_list.append((np.asarray(bb), bb_tups, sc, float(loss), aa_codes, atom_names))
    print("Avg Loss = {0:.2f}".format(np.mean(losses)))
    return coords_list


def fill_in_residue(resname, pred_coords, pred_names, bb_cords, reference_sidechains):
    """ Given an amino acid RESNAME that is partially predicted (only the atoms in PRED_NAMES are predicted),
        this function returns a list of coords that represents the complete amino acid structure."""
    missing = SC_DATA[resname]["missing"]
    align_target = SC_DATA[resname]["align_target"]
    align_mobile = SC_DATA[resname]["align_mobile"]

    if resname in ["ALA", "CYS", "GLY", "LYS", "MET", "SER"]:  # return the prev. predicted atoms if none are missing
        assert len(missing) == 0, "An atom that is fully predicted by the model has > 0 \"missing\" atoms."
        return list(zip(pred_names, pred_coords))

    if resname is "PRO":
        pred_coords = bb_cords
        pred_names = ["N", "CA", "C"]
    elif "CA" == align_target[0]:
        pred_coords = [bb_cords[1]] + pred_coords  # If target requires CA, add it to the list of predicted coords
        pred_names = ["CA"] + pred_names
    elif "CA" in align_target:
        raise Exception("CA found in target but not at position 0" + str(align_target) + " " + resname)

    # Load reference structures
    complete_target = reference_sidechains[resname].copy()
    complete_mobile = reference_sidechains[resname].copy()

    # Select relevant subsets of reference structures
    align_target_struct = complete_target.select("name " + " ".join(align_target))
    align_mobile_complete_struct = complete_mobile.select("name " + " ".join(align_mobile))
    align_mobile_struct = complete_mobile.select("name " + " ".join(set(align_mobile).intersection(set(align_target))))

    # Initialize target structure with predicted coordinates
    for an, at in zip(pred_names[-3:], align_target):
        assert an == at, "Lists are out of order" + resname + str(pred_names + [" "] + align_target)
    align_target_struct.setCoords(pred_coords[-3:])

    # Compute and apply transformation from mobile to target
    t = calcTransformation(align_mobile_struct, align_target_struct)
    t.apply(align_mobile_complete_struct)

    missing_coords = []

    for m in missing:
        c = align_mobile_complete_struct.select("name " + m).getCoords()[0]
        missing_coords.append((m, c))
    if resname is "PRO":
        return missing_coords
    elif "CA" in align_target:
        pred_coords = pred_coords[1:]
        pred_names = pred_names[1:]

    return list(zip(pred_names, pred_coords)) + missing_coords


def clean_multiple_coordsets(protein):
    """ Deletes all but the first coordinate set of a protein. """
    if len(protein.getCoordsets()) > 1:
        for i in range(len(protein.getCoordsets())):
            if i == 0:
                pass
            else:
                protein.delCoordset(-1)
    return protein


def set_backbone_coords(prot, chain_id, bb_coords, pdb_chain):
    # Set backbone atoms
    backbone = prot.select('protein and chain ' + chain_id + ' and name N CA C')
    assert backbone.getCoords().shape == bb_coords.shape, "Backbone shape mismatch for " + pdb_chain
    backbone.setCoords(bb_coords)
    # TODO Fill in oxygen position
    return backbone


def set_sidechain_coords(prot, aa_codes, bb_tups, sc_tups, atom_names, chain_id, ref_sidechains):
    """ Given a protein PDB selection object and amino acid"""
    # Set sidechain atoms
    assert len(aa_codes) == len(sc_tups) and len(sc_tups) == len(atom_names), "Shape mismatch for coord tuples."
    for sc_atomcoords, ans in zip(sc_tups, atom_names):
        assert len(sc_atomcoords) == len(ans)

    prot_sidechains = prot.select('protein and chain ' + chain_id)
    for res_num, res_code, res_coords, res_bb_coords, res_atom_names in zip(prot_sidechains.ca.getResnums(),
                                                                            aa_codes,
                                                                            sc_tups,
                                                                            bb_tups,
                                                                            atom_names):
        if res_code is "GLY":
            continue
        sidechain_coords = fill_in_residue(res_code, res_coords, res_atom_names, res_bb_coords, ref_sidechains)
        if res_num < 0:
            res_num_str = "`{0}`".format(str(res_num))
        else:
            res_num_str = str(res_num)
        this_sidechain = prot_sidechains.select('sidechain and resnum ' + res_num_str)
        if this_sidechain is None:
            print('this_sidechain is None')
            raise Exception("The sidechain could not be selected for " + res_code)
        for name, coord in sidechain_coords:
            this_sidechain.select("name " + name).setCoords(coord)
    return prot


def make_pdbs(id_coords_dict, outdir):
    """ Given a dictionary that maps PDB_ID -> pred_coordinate_tuple, this function parses the true PDB file and
        assigns coordinates to its atoms so that a PDB file can be generated."""
    os.makedirs(outdir, exist_ok=True)

    # Load reference sidechain structures
    ref_sidechains = {}
    for res in SC_DATA.keys():
        try:
            ref_sidechains[res] = parsePDB("data/amino_acid_substructures/" + res.lower() + ".pdb")
        except OSError:
            continue

    # Build PDBs
    for pdb_chain, data in id_coords_dict.items():
        bb_coords, bb_tups, sc_tups, loss, aa_codes, atom_names = data
        pdb_id = pdb_chain.split('_')[0]
        chain_id = pdb_chain.split("_")[1]
        print("Building", pdb_id, chain_id)

        prot = clean_multiple_coordsets(parsePDB(pdb_id))

        backbone = set_backbone_coords(prot, chain_id, bb_coords, pdb_chain)
        if args.backbone_only:
            writePDB(os.path.join(outdir, pdb_chain + '_l{0:.2f}.pdb'.format(loss)), backbone)
            continue

        prot = set_sidechain_coords(prot, aa_codes, bb_tups, sc_tups, atom_names, chain_id, ref_sidechains)
        all_sidechains = prot.select('protein and chain ' + chain_id + ' and sidechain')

        writePDB(os.path.join(outdir, pdb_chain + '_l{0:.2f}.pdb'.format(loss)), all_sidechains + backbone)


if __name__ == "__main__":
    np.random.seed(11)
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
    parser.add_argument("--pdb_dir", default="/home/jok120/pdb/", type=str, help="Path for ProDy-downloaded PDB files.")
    parser.add_argument("-bb", "--backbone_only", action="store_true", help="Only predict the protein backbone.")
    parser.add_argument("--reconstruct", action="store_true",
                        help="For debugging structure generation. Try to reconstruct the true protein structure.")
    parser.add_argument("--include_truth", action="store_true", help="Include the true structure in the PDB file.")
    args = parser.parse_args()
    pathPDBFolder(args.pdb_dir)
    if args.reconstruct:
        print("Attempting to reconstruct real structures.")

    # Load model
    device = torch.device('cpu')
    args, the_model = load_model(args)

    # Acquire seqs and angles to predict / compare from
    data_loader, ids = get_data_loader(torch.load(args.data), args.dataset, n=args.n)

    # Make predictions as coordinates
    coords_list = make_predictions(the_model, data_loader, ids)
    if args.include_truth:
        coords_list += make_predictions(the_model, data_loader, ids, build_true=True)
        ids += [i + "_TRUE_" for i in ids]
    id_coords_dict = {k: v for k, v in zip(ids, coords_list)}

    # Make PDB files from coords
    make_pdbs(id_coords_dict, args.outdir)
