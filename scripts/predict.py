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


def get_data_loader(data_dict, dataset, n=3):
    to_predict = np.random.choice(data_dict[dataset]["ids"], n)  # ["2NLP_D", "3ASK_Q", "1SZA_C"]
    actual_order = []
    seqs = []
    angs = []
    for i, prot in enumerate(data_dict[dataset]["ids"]):
        if prot.upper() in to_predict:
            seqs.append(data_dict[dataset]["seq"][i])
            angs.append(data_dict[dataset]["ang"][i])
            actual_order.append(prot)
    assert len(seqs) == n and len(angs) == n

    data_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=seqs,
            angs=angs),
        num_workers=2,
        batch_size=1,
        collate_fn=paired_collate_fn,
        shuffle=False)
    return data_loader, actual_order


def make_predictions(the_model, data_loader):
    coords_list = []
    losses = []
    norm_losses = []

    # TODO: make batch_level predictions?
    with torch.no_grad():
        for batch in tqdm(data_loader, mininterval=2,
                          desc=' - (Evaluation ', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = batch
            gold = tgt_seq[:]

            # forward
            pred = the_model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, loss_norm = cal_loss(pred, gold, src_seq, device, combined=False)
            losses.append(loss)
            norm_losses.append(loss_norm)

            pred, gold = inverse_trig_transform(pred), inverse_trig_transform(gold)
            pred, gold = copy_padding_from_gold(pred, gold, torch.device('cpu'))

            # print('Loss: {0:.2f}, NLoss: {1:.2f}, Predshape: {2}'.format(float(loss), float(loss_norm), pred.shape))
            all, bb, sc = struct.generate_coords(pred[0], pred.shape[1], src_seq[0], torch.device('cpu'),
                                                 return_tuples=True)
            coords_list.append((np.asarray(bb), np.asarray(sc), float(loss), float(loss_norm)))
    print("Avg Loss = {0:.2f}, Avg NLoss = {1:.2f}".format(np.mean(losses), np.mean(norm_losses)))
    return coords_list


def make_pdbs(id_coords_dict, outdir):
    os.makedirs(outdir, exist_ok=True)
    for key in id_coords_dict.keys():
        bb_coords, sc_coords, loss, loss_norm = id_coords_dict[key]
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

        backbone = prot.select('protein and chain ' + chain_id + ' and name N CA C')

        if not backbone.getCoords().shape == bb_coords.shape:
            # there is an error in the process!!
            print('Error! Shape mismatch for ' + str(key))
        else:
            backbone.setCoords(bb_coords)
            writePDB(os.path.join(outdir, key + '_nl{0:.2f}.pdb'.format(loss_norm)), backbone)


if __name__ == "__main__":
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
                        help="Which dataset within the data file to predict on (one of {train, valid, test, all}).")
    parser.add_argument("-n", type=int, default=5, required=False,
                        help="How many items to randomly predict from dataset.")

    args = parser.parse_args()

    # Load model
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

    # Aquire seqs and angles to predict / compare from
    data_loader, ids = get_data_loader(torch.load(args.data), args.dataset, n=args.n)

    # Make predictions as coordinates
    coords_list = make_predictions(the_model, data_loader)
    id_coords_dict = {k: v for k, v in zip(ids, coords_list)}

    # Make PDB files from coords
    make_pdbs(id_coords_dict, args.outdir)
