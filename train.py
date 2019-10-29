"""
Primary script for training models to predict protein structure from amino
acid sequence.

    Author: Jonathan King
    Date: 10/25/2019
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from dataset import prepare_dataloaders
from losses import drmsd_loss_from_angles, mse_over_angles, combine_drmsd_mse
from models.transformer.Models import Transformer, MISSING_CHAR
from models.transformer.Optim import ScheduledOptim
from log import *
from models.nmt_models import EncoderOnlyTransformer

wandb.init(project="protein-transformer", entity="koes-group")

def train_epoch(model, training_data, optimizer, device, args, log_writer, metrics):
    """
    One complete training epoch.
    """
    model.train()
    metrics = reset_metrics_for_epoch(metrics, "train")
    n_batches = 0.0
    pbar = tqdm(training_data, mininterval=2, leave=False) if not args.cluster else training_data

    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        src_seq, src_pos_enc, tgt_ang, tgt_pos_enc, tgt_crds, tgt_crds_enc = map(lambda x: x.to(device), batch)
        if args.skip_missing_res_train and torch.isnan(tgt_ang).all(dim=-1).any().byte():
            continue
        tgt_ang_no_nan = tgt_ang.clone().detach()
        tgt_ang_no_nan[torch.isnan(tgt_ang_no_nan)] = MISSING_CHAR
        # We don't provide the entire output sequence to the model because it will be given t-1 and should predict t
        pred = model(src_seq.argmax(dim=-1))
        pred_coords, d_loss, ln_d_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, device)
        d_loss, ln_d_loss = d_loss.to('cpu'), ln_d_loss.to('cpu')
        m_loss = mse_over_angles(pred, tgt_ang).to('cpu')
        c_loss = combine_drmsd_mse(ln_d_loss, m_loss, w=0.5)
        loss = c_loss if args.combined_loss else ln_d_loss
        loss.backward()

        # Clip gradients
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # update parameters
        optimizer.step()

        # record performance metrics
        do_train_batch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, src_seq, loss, optimizer, args, log_writer,
                               pbar, START_TIME, pred_coords, tgt_crds[-1], step)
        n_batches += 1

    metrics = update_metrics_end_of_epoch(metrics, "train", n_batches)

    return metrics


def eval_epoch(model, validation_data, device, args, metrics, mode="valid"):
    """
    One compete evaluation epoch.
    """
    model.eval()
    metrics = reset_metrics_for_epoch(metrics, mode)
    n_batches = 0.0
    pbar = tqdm(validation_data, mininterval=2, leave=False) if not args.cluster else validation_data

    with torch.no_grad():
        for batch in pbar:
            src_seq, src_pos_enc, tgt_ang, tgt_pos_enc, tgt_crds, tgt_crds_enc = map(lambda x: x.to(device), batch)
            pred = model.predict(src_seq.argmax(dim=-1))
            pred_coords, d_loss, ln_d_loss, r_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, device, return_rmsd=True)
            m_loss = mse_over_angles(pred, tgt_ang).to('cpu')
            c_loss = combine_drmsd_mse(ln_d_loss, m_loss)
            do_eval_epoch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, r_loss, src_seq, args, pbar, mode)
            n_batches += 1

    metrics = update_metrics_end_of_epoch(metrics, mode, n_batches)
    return metrics


def train(model, metrics, training_data, validation_data, test_data, optimizer, device, args, log_writer):
    """
    Model training control loop.
    """
    for epoch_i in range(START_EPOCH, args.epochs):
        print(f'[ Epoch {epoch_i} ]')

        # Train epoch
        start = time.time()
        metrics = train_epoch(model, training_data, optimizer, device, args, log_writer, metrics)
        if args.eval_train:
           metrics = eval_epoch(model, training_data, device, args, metrics, mode="train")
        print_end_of_epoch_status("train", (start, metrics))
        log_batch(log_writer, metrics, START_TIME, mode="train", end_of_epoch=True)

        # Valid epoch
        if not args.train_only:
            start = time.time()
            metrics = eval_epoch(model, validation_data, device, args, metrics)
            print_end_of_epoch_status("valid", (start, metrics))
            log_batch(log_writer, metrics, START_TIME, mode="valid", end_of_epoch=True)

        # Checkpointing
        try:
            metrics = update_loss_trackers(args, epoch_i, metrics)
        except EarlyStoppingCondition:
            break
        checkpoint_model(args, optimizer, model, metrics, epoch_i)

    # Test Epoch
    if not args.train_only:
        start = time.time()
        metrics = eval_epoch(model, test_data, device, args, metrics, mode="test")
        print_end_of_epoch_status("test", (start, metrics))
        log_batch(log_writer, metrics, START_TIME, mode="test", end_of_epoch=True)


def checkpoint_model(args, optimizer, model, metrics, epoch_i):
    """
    Records model state according to a checkpointing policy. Defaults to best
    validation set performance. Returns True iff model was saved.
    """
    cur_loss, loss_history = metrics["loss_to_compare"], metrics["losses_to_compare"]
    if args.save_mode == 'all' or len(loss_history) == 1 or cur_loss < min(loss_history[:-1]):
        model_state_dict = model.state_dict()
        checkpoint = {
            'model_state_dict': model_state_dict,
            'settings': args,
            'epoch': epoch_i,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': cur_loss,
            'metrics': metrics,
            'elapsed_time':time.time() - START_TIME}
    else:
        return False

    if args.save_mode == 'all':
        chkpt_file_name = args.chkpt_path + "_epoch-{0}_vloss-{1}.chkpt".format(
            epoch_i, cur_loss)
    else:
        chkpt_file_name = args.chkpt_path + "_best.chkpt"

    torch.save(checkpoint, chkpt_file_name)
    wandb.save(chkpt_file_name)
    print('\r    - [Info] The checkpoint file has been updated.')
    wandb.run.summary["best_validation_loss"] = cur_loss
    wandb.run.summary["avg_evaluation_speed"] = np.mean(
        metrics["valid"]["speed-history"])
    wandb.run.summary["avg_training_speed"] = np.mean(
        metrics["train"]["speed-history"])

    return True


def load_model(model, optimizer, args):
    """
    Given a model, its optimizer, and the program's arguments, resumes model
    training if the user has not specified otherwise. Assumes model was saved
    the 'best' mode.
    """
    global START_EPOCH
    global START_TIME
    chkpt_file_name = args.chkpt_path + "_best.chkpt"

    # Try to load the model checkpoint, if it exists
    if os.path.exists(chkpt_file_name) and not args.restart:
        print(f"[Info] Attempting to load model from {chkpt_file_name}.")
    else:
        return model, optimizer, False, init_metrics(args)
    checkpoint = torch.load(chkpt_file_name)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("[Info] Error loading model.")
        print(e)
        exit(1)

    # Load the optimizer state by default
    if not args.restart_opt:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    START_EPOCH = checkpoint['epoch'] + 1
    START_TIME -= checkpoint['elapsed_time']
    print(f"[Info] Resuming model training from end of Epoch {checkpoint['epoch']}. Previous validation loss"
          f" = {checkpoint['loss']:.4f}.")
    return model, optimizer, True, checkpoint['metrics']


def main():
    """
    Argument parsing, model loading, and model training.
    """
    global LOGFILEHEADER
    global START_EPOCH
    global START_TIME
    START_EPOCH = 0
    START_TIME = time.time()

    torch.set_printoptions(precision=5, sci_mode=False)
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument('data', help="Path to training data.")
    parser.add_argument("name", type=str, help="The model name.")

    # Training parameters
    parser.add_argument("-lr", "--learning_rate", type=float, default=1 * (10 ** -3))
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument("-b", '--batch_size', type=int, default=8)
    parser.add_argument('-es', '--early_stopping', type=int, default=None,
                        help="Stops if training hasn't improved in X epochs")
    parser.add_argument('-nws', '--n_warmup_steps', type=int, default=10000)
    parser.add_argument('-cg', '--clip', type=float, default=None)
    parser.add_argument('-cl', '--combined_loss', action='store_true',
                        help="Use a loss that combines (quasi-equally) DRMSD and MSE.")
    parser.add_argument('--train_only', action='store_true',
                        help="Train, validation, and testing sets are the same. Only report train accuracy.")
    parser.add_argument('--lr_scheduling', action='store_true', help='Use learning rate scheduling as described in" + '
                                                                     '"original paper.')
    parser.add_argument('--without_angle_means', action='store_true',
                        help="Do not initialize the model with pre-computed angle means.")
    parser.add_argument('--eval_train', action='store_true',
                        help="Perform an evaluation of the entire training set after a training epoch.")
    parser.add_argument('-opt', '--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument("-fctf", "--fraction_complete_tf", type=float, default=1,
                        help="Fraction of the time to use teacher forcing for every timestep of the batch. Model trains"
                             "fastest when this is 1.")
    parser.add_argument("-fsstf", "--fraction_subseq_tf", type=float, default=1,
                        help="Fraction of the time to use teacher forcing on a per-timestep basis.")
    parser.add_argument("--skip_missing_res_train", action="store_true",
                        help="When training, skip over batches that have missing residues. This can make training"
                             "faster if using teacher forcing.")
    parser.add_argument("--repeat_train", type=int, default=1, help="Duplicate the training set X times.")

    # Model parameters
    parser.add_argument('-dm', '--d_model', type=int, default=512)
    parser.add_argument('-dih', '--d_inner_hid', type=int, default=2048)
    parser.add_argument('-dk', '--d_k', type=int, default=64)
    parser.add_argument('-dv', '--d_v', type=int, default=64)
    parser.add_argument('-nh', '--n_head', type=int, default=8)
    parser.add_argument('-nl', '--n_layers', type=int, default=6)
    parser.add_argument('-do', '--dropout', type=float, default=0)
    parser.add_argument('--postnorm', action='store_true', help="Use post-layer normalization, as depicted in the "
                        "original figure for the Transformer model. May not train as well as pre-layer normalization.")

    # Saving args
    parser.add_argument('--log_structure_step', type=int, default=10)
    parser.add_argument('--log_wandb_step', type=int, default=1)
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--cluster', action='store_true', help="Set of parameters to facilitate training on a remote" +
                                                               " cluster. Limited I/O, etc.")
    parser.add_argument('--restart', action='store_true', help="Does not resume training.")
    parser.add_argument('--restart_opt', action='store_true', help="Resumes training but does not load the optimizer"
                                                                   "state. ")

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.buffering_mode = 1
    LOGFILEHEADER = prepare_log_header(args)
    if args.save_mode == "all" and not args.restart:
        print("You cannot resume this model because it was saved with mode 'all'.")
        exit(1)

    # Load dataset
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings']["max_len"]
    training_data, validation_data, test_data = prepare_dataloaders(data, args)

    # Prepare model
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = EncoderOnlyTransformer(nlayers=args.n_layers,
                                   nhead=args.n_head,
                                   dmodel=args.d_model,
                                   dff=args.d_inner_hid,
                                   max_seq_len=500,
                                   dropout=args.dropout).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                               betas=(0.9, 0.98), eps=1e-09, lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=args.learning_rate)
    if args.lr_scheduling:
        optimizer = ScheduledOptim(optimizer, args.d_model, args.n_warmup_steps, simple=False)

    # Prepare log and checkpoint files
    args.chkpt_path = "./data/checkpoints/" + args.name
    os.makedirs("./data/checkpoints", exist_ok=True)
    args.log_file = "./data/logs/" + args.name + '.train'
    print('[Info] Training performance will be written to file: {}'.format(args.log_file))
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    model, optimizer, resumed, metrics = load_model(model, optimizer, args)
    if resumed:
        log_f = open(args.log_file, 'a', buffering=args.buffering_mode)
    else:
        log_f = open(args.log_file, 'w', buffering=args.buffering_mode)
        log_f.write(LOGFILEHEADER)
    log_writer = csv.writer(log_f)
    wandb.watch(model, "all")
    wandb.config.update(args)
    if type(data["date"]) == set:
        wandb.config.update({"data_creation_date": next(iter(data["date"]))})
    else:
        wandb.config.update({"data_creation_date": data["date"]})
    print(args, "\n")

    # Begin training
    train(model, metrics, training_data, validation_data, test_data, optimizer, device, args, log_writer)
    log_f.close()


if __name__ == '__main__':
    main()
