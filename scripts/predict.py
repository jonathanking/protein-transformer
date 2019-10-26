"""" This script will take a trained model and a target dataset to predict on
and make predictions that can be viewed as PDB files. It's current mode of
action is to download the corresponding PDB file for a protein and replace
its coordinates with the matching ones from the model. """

import argparse
import os
import sys

sys.path.append("/home/jok120/protein-transformer/")
sys.path.append("/home/jok120/protein-transformer/scripts")

import torch
from tqdm import tqdm
from prody import *
import numpy as np
from os.path import basename, splitext

import torch.utils.data
from dataset import ProteinDataset, paired_collate_fn, paired_collate_fn_with_len
from protein.Structure import generate_coords_with_tuples
from losses import inverse_trig_transform, copy_padding_from_gold, drmsd_loss_from_angles, mse_over_angles, combine_drmsd_mse
from protein.Sidechains import SC_DATA
from models.rnn import MyRNN
from proteinnet2pytorch import get_chain_from_proteinnetid

VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]


def load_model(args):
    """
    Given user-supplied arguments such as a model checkpoint, loads and
    returns the specified transformer model. If the data to predict is not
    specified, the original file used during training will be re-used.
    """
    # TODO remove try/except clause for loading model
    chkpt = torch.load(args.model_chkpt, map_location=device)
    model_args = chkpt['settings']
    model_state = chkpt['model_state_dict']
    if args.data is None:
        args.data = model_args.data

    try:
        if model_args.rnn is None:
            model_args.rnn = False
            args.rnn = False
    except AttributeError:
        model_args.rnn = False
        args.rnn = False

    if not args.rnn:
        the_model = models.transformer.Models.Transformer(model_args,
                                                          d_k=model_args.d_k,
                                                          d_v=model_args.d_v,
                                                          d_model=model_args.d_model,
                                                          d_inner=model_args.d_inner_hid,
                                                          n_layers=model_args.n_layers,
                                                          n_head=model_args.n_head,
                                                          dropout=model_args.dropout)
    else:
        latent_dim, n_layers, bidi = model_args.d_model, model_args.n_layers, True
        the_model = MyRNN(model_args, latent_dim, num_layers=n_layers, bidirectional=bidi, device=device)
    the_model.load_state_dict(model_state)
    return args, the_model


def get_data_loader(args, data_subset, n):
    """
    Given a subset of a dataset as a python dictionary file to make
    predictions from, this function selects n items at random from that
    dataset to predict. It then returns a DataLoader for those items,
    along with a list of ids.
    """
    if not args.rnn:
        collate = paired_collate_fn
    else:
        collate = paired_collate_fn_with_len
    if n is 0:
        train_loader = torch.utils.data.DataLoader(
            ProteinDataset(
                seqs=data_subset['seq'],
                crds=data_subset['crd'],
                angs=data_subset['ang'],
                ),
            num_workers=2,
            batch_size=1,
            collate_fn=collate,
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
        collate_fn=collate,
        shuffle=False)
    return data_loader, ids


def make_predictions(args, the_model, data_loader, pdb_ids, build_true=False):
    """
    Given a loaded transformer model, and a dataloader of items to predict,
    this model returns a list of tuples. Each tuple is contains (backbone
    coord. matrix, sidechain coord. matrix, loss, nloss) for a single item.
    """
    coords_list = []
    losses = {"drmsd": [],
              "mse": [],
              "rmsd": [],
              "combined": [],
              "rmsd-backbone": [],
              "rmsd-allatom": []}

    with torch.no_grad():
        for pdb_id, batch in zip(pdb_ids, tqdm(data_loader, mininterval=2, desc=' - (Evaluation ', leave=False)):
            # prepare data
            if args.rnn:
                lens, src_seq, src_pos_enc, tgt_ang, tgt_pos_enc, tgt_crds, tgt_crds_enc = map(lambda x: x.to(device),
                                                                                               batch)
            else:
                src_seq, src_pos_enc, tgt_ang, tgt_pos_enc, tgt_crds, tgt_crds_enc = map(lambda x: x.to(device), batch)

            # forward
            if args.reconstruct or build_true:
                pred = tgt_ang
            elif args.rnn:
                pred = the_model(src_seq, lens)
            else:
                tgt_ang_no_nan = tgt_ang.clone().detach()
                tgt_ang_no_nan[torch.isnan(tgt_ang_no_nan)] = 0
                pred = the_model(src_seq, src_pos_enc, tgt_ang_no_nan, tgt_pos_enc)

            # TODO return backbone vs sidechain losses
            try:
                d_loss, r_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, device, return_rmsd=True)
                m_loss = mse_over_angles(pred, tgt_ang)
            except (AssertionError, ValueError):
                continue
            c_loss = combine_drmsd_mse(d_loss, m_loss)
            losses["drmsd"].append(d_loss.item())
            losses["mse"].append(m_loss.item())
            losses["rmsd"].append(r_loss)
            losses["combined"].append(c_loss.item())


            pred, gold = inverse_trig_transform(pred), inverse_trig_transform(tgt_ang)
            pred, gold = copy_padding_from_gold(pred, gold, torch.device('cpu'))

            all, bb, bb_tups, sc, aa_codes, atom_names = generate_coords_with_tuples(pred[0], pred.shape[1], src_seq[0],
                                                                                     torch.device('cpu'))
            coords_list.append((np.asarray(bb), bb_tups, sc, float(r_loss), aa_codes, atom_names, pred[0]))
    print("Avg Loss = {0:.2f}".format(np.mean(losses["drmsd"])))
    torch.save(losses, os.path.join(args.outdir, f"{splitext(basename(args.model_chkpt))[0]}_{args.dataset}_trans-evalution.tch"))
    return coords_list


def fill_in_residue(resname, pred_coords, pred_names, bb_cords, reference_sidechains, angles):
    """
    Given an amino acid RESNAME that is partially predicted (only the atoms
    in PRED_NAMES are predicted), this function returns a list of coords that
    represents the complete amino acid structure.
    """
    missing = SC_DATA[resname]["missing"]
    align_target = SC_DATA[resname]["align_target"]
    align_mobile = SC_DATA[resname]["align_mobile"]

    if resname in ["ALA", "CYS", "GLY", "LYS", "MET", "SER"] + ["ILE", "VAL", "LEU", "THR"]:
        # return the prev. predicted atoms if none are missing
        return list(zip(pred_names, pred_coords))

    if resname is "PRO":
        pred_coords = bb_cords[:2] + pred_coords
        pred_names = ["N", "CA", "CB"]
    elif "CA" == align_target[0]:
        pred_coords = [bb_cords[1]] + pred_coords  # If target requires CA, add it to the list of predicted coords
        pred_names = ["CA"] + pred_names
    elif "CA" in align_target and resname is not "PRO":
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
        pred_coords = [pred_coords[2]]
        pred_names = [pred_names[2]]
    elif "CA" in align_target:
        pred_coords = pred_coords[1:]
        pred_names = pred_names[1:]

    return list(zip(pred_names, pred_coords)) + missing_coords


def clean_multiple_coordsets(protein):
    """
    Deletes all but the first coordinate set of a protein.
    """
    if len(protein.getCoordsets()) > 1:
        for i in range(len(protein.getCoordsets())):
            if i == 0:
                pass
            else:
                protein.delCoordset(-1)
    return protein


def set_backbone_coords(prot, chain_id, bb_coords, pdb_chain, peptide_bond):
    # Set backbone atoms
    backbone = prot.select('protein and chain ' + chain_id + ' and name N CA C')
    assert backbone.getCoords().shape == bb_coords.shape, "Backbone shape mismatch for " + pdb_chain
    backbone.setCoords(bb_coords)
    backbone = prot.select('protein and chain ' + chain_id + ' and backbone')

    # Position oxygen atoms by aligning a pre-computed peptide bond to each residue.
    # TODO Nerf is probably more efficient at placing oxygen
    prev_res = None
    prev_ca_c = None
    for res in prot.select('protein and chain ' + chain_id).getHierView().iterResidues():
        if prev_res is None:
            prev_ca_c = res.select("name CA").getCoords()[0], res.select("name C").getCoords()[0]
            prev_res = res
            continue
        ca, c, n = prev_ca_c[0], prev_ca_c[1], res.select("name N").getCoords()[0]
        target_coords = np.array([ca, c, n])
        mobile_coords_complete = peptide_bond.copy().getCoords()
        assert list(peptide_bond.getNames()) == ["CA", "C", "O", "N"]
        mobile_coords_align = peptide_bond.copy().select("name CA C N")
        t = calcTransformation(mobile_coords_align.getCoords(), target_coords)
        mobile_coords_aligned = t.apply(mobile_coords_complete)
        prev_res.select("name O").setCoords(mobile_coords_aligned[2])

        prev_ca_c = res.select("name CA").getCoords()[0], res.select("name C").getCoords()[0]
        prev_res = res

    return backbone


def set_sidechain_coords(prot, aa_codes, bb_tups, sc_tups, atom_names, chain_id, ref_sidechains, angles):
    # Set sidechain atoms
    assert len(aa_codes) == len(sc_tups) and len(sc_tups) == len(atom_names), "Shape mismatch for coord tuples."
    for sc_atomcoords, ans in zip(sc_tups, atom_names):
        assert len(sc_atomcoords) == len(ans)

    prot_sidechains = prot.select('protein and chain ' + chain_id)
    for res_num, res_code, res_coords, res_bb_coords, res_atom_names, ang in zip(prot_sidechains.ca.getResnums(),
                                                                                 aa_codes,
                                                                                 sc_tups,
                                                                                 bb_tups,
                                                                                 atom_names, angles):
        if res_code is "GLY":
            continue
        sidechain_coords = fill_in_residue(res_code, res_coords, res_atom_names, res_bb_coords, ref_sidechains, ang)
        if res_num < 0:
            res_num_str = "`{0}`".format(str(res_num))
        else:
            res_num_str = str(res_num)
        this_sidechain = prot_sidechains.select('sidechain and resnum ' + res_num_str)
        if this_sidechain is None:
            # print('this_sidechain is None')
            # raise Exception("The sidechain could not be selected for " + res_code)
            continue
        for name, coord in sidechain_coords:
            s = this_sidechain.select("name " + name)
            if s: s.setCoords(coord)
    return prot


def make_pdbs(id_coords_dict, outdir):
    """
    Given a dictionary that maps PDB_ID -> pred_coordinate_tuple,
    this function parses the true PDB file and assigns coordinates to its
    atoms so that a PDB file can be generated.
    """
    
    # Load reference sidechain structures
    ref_sidechains = {}
    for res in SC_DATA.keys():
        try:
            ref_sidechains[res] = parsePDB("data/amino_acid_substructures/" + res.lower() + ".pdb")
        except OSError:
            continue
    peptide_bond = parsePDB("data/amino_acid_substructures/peptide_bond.pdb")
    coords_complete = {}

    # Build PDBs
    for pdb_chain, data in id_coords_dict.items():
        bb_coords, bb_tups, sc_tups, loss, aa_codes, atom_names, angles = data
        print("Building", pdb_chain)
        # pdb_id = pdb_chain.split('_')[0]
        # chain_id = pdb_chain.split("_")[1]
        chain_id = pdb_chain.split("_")[-1]


        prot = get_chain_from_proteinnetid(pdb_chain)
        if not prot:
            continue
        try:
            backbone = set_backbone_coords(prot, chain_id, bb_coords, pdb_chain, peptide_bond)
        except AssertionError:
            print(f'backbone mismatch for {pdb_chain}.')
            continue
        if args.backbone_only:
            writePDB(os.path.join(outdir, pdb_chain + '_l{0:.2f}.pdb'.format(loss)), backbone)
            continue
        try:
            prot = set_sidechain_coords(prot, aa_codes, bb_tups, sc_tups, atom_names, chain_id, ref_sidechains, angles)
        except AttributeError as e:
            print(e)
            continue
        all_sidechains = prot.select('protein and chain ' + chain_id + ' and sidechain')
        all_atoms = all_sidechains + backbone
        for k in coords_complete.keys():
           if pdb_chain[:6] in k and "TRUE" in pdb_chain:
               t = calcTransformation(all_atoms, coords_complete[k])
               all_atoms = t.apply(all_atoms)
               break
        coords_complete[pdb_chain] = all_atoms

        writePDB(os.path.join(outdir, pdb_chain + '_l{0:.2f}.pdb'.format(loss)), all_atoms)


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
    parser.add_argument("-dataset", type=str, choices=["train", "test", "all"] + ["valid-" + str(i) for i in VALID_SPLITS], default="test",
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
    os.makedirs(args.outdir, exist_ok=True)

    if args.reconstruct:
        print("Attempting to reconstruct real structures.")

    # Load model
    device = torch.device('cpu')
    args, the_model = load_model(args)

    # Acquire seqs and angles to predict / compare from
    data = torch.load(args.data)
    if 'valid' in args.dataset:
        v, num = args.dataset.split("-")
        data_subset = data[v][int(num)]
    else:
        data_subset = data[args.dataset]
    data_loader, ids = get_data_loader(args, data_subset, n=args.n)

    # Make predictions as coordinates
    coords_list = make_predictions(args, the_model, data_loader, ids)
    if args.include_truth:
        coords_list += make_predictions(args, the_model, data_loader, ids, build_true=True)
        ids += [i + "_TRUE_" for i in ids]
    id_coords_dict = {k: v for k, v in zip(ids, coords_list)}

    # Make PDB files from coords
    make_pdbs(id_coords_dict, args.outdir)
