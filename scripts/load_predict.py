import numpy as np
import torch

np.random.seed(7)
from tqdm import tqdm
import sys
import pickle

if len(sys.argv) != 4:
    print('Improper usage!')
    print('load_and_predict.py <model> <outpickle> <dataset_to_predict{train/test}>')
    exit(1)

model_path = sys.argv[1]
out_path = sys.argv[2]
dataset = sys.argv[3]
loss_out_path = out_path[:-6] + "_drmsd.npy"
norm_loss_out_path = out_path[:-6] + "_ndrmsd.npy"

device = torch.device('cpu')
chkpt = torch.load(model_path, map_location=device)
opt = chkpt['settings']
model_state = chkpt['model']

import transformer.Models
import torch.utils.data
from dataset import ProteinDataset, paired_collate_fn
import protein.Structure as struct
from train import cal_loss
from losses import inverse_trig_transform, copy_padding_from_gold


the_model = transformer.Models.Transformer(opt.max_token_seq_len,
                                           d_k=opt.d_k,
                                           d_v=opt.d_v,
                                           d_model=opt.d_model,
                                           d_inner=opt.d_inner_hid,
                                           n_layers=opt.n_layers,
                                           n_head=opt.n_head,
                                           dropout=opt.dropout)
the_model.load_state_dict(model_state)

data = torch.load(opt.data)

to_predict = np.random.choice(data[dataset]["ids"], 3)  # ["2NLP_D", "3ASK_Q", "1SZA_C"]
actual_order = []
seqs = []
angs = []
for i, prot in enumerate(data[dataset]["ids"]):
    if prot.upper() in to_predict:
        seqs.append(data[dataset]["seq"][i])
        angs.append(data[dataset]["ang"][i])
        actual_order.append(prot)
assert len(seqs) == 3 and len(angs) ==3

data_loader = torch.utils.data.DataLoader(
    ProteinDataset(
        seqs=seqs,#data[dataset]['seq'],
        angs=angs),#data[dataset]['ang']),
    num_workers=2,
    batch_size=1,
    collate_fn=paired_collate_fn,
    shuffle=False)

cords_list = []
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
        try:
            pred = the_model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, loss_norm = cal_loss(pred, gold, src_seq, device, combined=False)
            losses.append(loss)
            norm_losses.append(loss_norm)
        except:
            continue

        pred, gold = inverse_trig_transform(pred), inverse_trig_transform(gold)
        pred, gold = copy_padding_from_gold(pred, gold, torch.device('cpu'))

        # pad_loc = int(np.argmax((gold == 0).sum(dim=-1)))
        # if pad_loc is 0:
        #     pad_loc = gold.shape[1]

        print('Loss: {0:.2f}, NLoss: {1:.2f}, Predshape: {2}'.format(float(loss), float(loss_norm), pred.shape))
        all, bb, sc = struct.generate_coords(pred[0], pred.shape[1], src_seq[0], torch.device('cpu'),
                                             return_tuples=True)
        cords_list.append((np.asarray(bb), np.asarray(sc), float(loss), float(loss_norm)))


    d = {}
    for key, val in zip(actual_order, cords_list):
        d[key] = val

    with open(out_path, 'wb') as f:
        pickle.dump(d, f)

    with open(loss_out_path, "wb") as f:
        np.save(f, np.array(losses))

    with open(norm_loss_out_path, "wb") as f:
        np.save(f, np.array(norm_losses))
