"""
Primary script for training models to predict protein structure from amino
acid sequence.

    Author: Jonathan King
    Date: 10/25/2019
"""

import argparse
import csv
import os
import random

import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import wandb

from dataset import prepare_dataloaders
from losses import drmsd_loss_from_angles, mse_over_angles, combine_drmsd_mse
from models.transformer.Transformer import Transformer
from models.transformer.Optimizer import ScheduledOptim
from log import *
from models.encoder_only import EncoderOnlyTransformer
from protein.Sidechains import NUM_PREDICTED_ANGLES


def train_epoch(model, training_data, optimizer, device, args, log_writer, metrics):
    """
    One complete training epoch.
    """
    model.train()
    metrics = reset_metrics_for_epoch(metrics, "train")
    n_batches = 0
    batch_iter = tqdm(training_data, leave=False, unit="batch", dynamic_ncols=True) if not args.cluster else training_data

    for step, batch in enumerate(batch_iter):
        optimizer.zero_grad()
        src_seq, tgt_ang, tgt_crds = map(lambda x: x.to(device), batch)
        if args.skip_missing_res_train and torch.isnan(tgt_ang).all(dim=-1).any().byte():
            continue
        pred = model(src_seq, tgt_ang)
        loss, d_loss, ln_d_loss, m_loss, c_loss, pred_coords = get_losses(args, step, device, pred, tgt_ang, tgt_crds, src_seq)
        loss.backward()

        # Clip gradients
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # Update parameters
        optimizer.step()

        # Record performance metrics
        do_train_batch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, src_seq, loss, optimizer, args,
                               log_writer, batch_iter, START_TIME, pred_coords, tgt_crds[-1], step)
        n_batches += 1

    metrics = update_metrics_end_of_epoch(metrics, "train", n_batches)

    return metrics


def get_losses(args, step, device, pred, tgt_ang, tgt_crds, src_seq):
    """
    Returns the computed losses/metrics for a batch. The variable 'loss'
    will differ depending on the loss the user requested to train on.
    """
    m_loss = mse_over_angles(pred, tgt_ang)

    if args.loss == "ln-drmsd":
        pred_coords, d_loss, ln_d_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, torch.device('cpu'))
        c_loss = combine_drmsd_mse(ln_d_loss.to(device), m_loss, w=0.5)
        loss = ln_d_loss
    elif args.loss == "mse" and step % args.log_structure_step != 0:
        # Other losses are not computed for computational efficiency.
        pred_coords, d_loss, ln_d_loss = None, torch.tensor(0), torch.tensor(0)
        c_loss = torch.tensor(0)
        loss = m_loss
    elif args.loss == "mse" and step % args.log_structure_step == 0:
        pred_coords, d_loss, ln_d_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, torch.device('cpu'))
        c_loss = combine_drmsd_mse(ln_d_loss.to(device), m_loss, w=0.5)
        loss = m_loss
    elif args.loss == "drmsd":
        pred_coords, d_loss, ln_d_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, torch.device('cpu'))
        c_loss = combine_drmsd_mse(ln_d_loss.to(device), m_loss, w=0.5)
        loss = d_loss
    else:
        pred_coords, d_loss, ln_d_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, torch.device('cpu'))
        c_loss = combine_drmsd_mse(ln_d_loss.to(device), m_loss, w=0.5)
        loss = c_loss

    return loss, d_loss, ln_d_loss, m_loss, c_loss, pred_coords


def eval_epoch(model, validation_data, device, args, metrics, mode="valid"):
    """
    One compete evaluation epoch.
    """
    model.eval()
    metrics = reset_metrics_for_epoch(metrics, mode)
    n_batches = 0.0
    batch_iter = tqdm(validation_data, mininterval=.5, leave=False) if not args.cluster else validation_data

    with torch.no_grad():
        for batch in batch_iter:
            src_seq, tgt_ang, tgt_crds = map(lambda x: x.to(device), batch)
            pred = model(src_seq, tgt_ang)
            pred_coords, d_loss, ln_d_loss, r_loss = drmsd_loss_from_angles(pred, tgt_crds, src_seq, torch.device("cpu"), return_rmsd=True)
            m_loss = mse_over_angles(pred, tgt_ang)
            c_loss = combine_drmsd_mse(ln_d_loss.to(device), m_loss)
            do_eval_epoch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, r_loss, src_seq, args, batch_iter, mode)
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
    if len(loss_history) == 1 or cur_loss < min(loss_history[:-1]):
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


def make_model(args, device):
    """
    Returns requested model architecture. Currently only enc-only and enc-dec
    are supported.
    """
    if args.model == "enc-only":
        model = EncoderOnlyTransformer(nlayers=args.n_layers,
                                       nhead=args.n_head,
                                       dmodel=args.d_model,
                                       dff=args.d_inner_hid,
                                       max_seq_len=MAX_SEQ_LEN,
                                       dropout=args.dropout,
                                       vocab=VOCAB,
                                       angle_mean_path=args.angle_mean_path)
    elif args.model == "enc-dec":
        model = Transformer(dm=args.d_model,
                            dff=args.d_inner_hid,
                            din=len(VOCAB),
                            dout=NUM_PREDICTED_ANGLES * 2,
                            n_heads=args.n_head,
                            n_enc_layers=args.n_layers,
                            n_dec_layers=args.n_layers,
                            max_seq_len=MAX_SEQ_LEN,
                            pad_char=VOCAB.pad_id,
                            missing_coord_filler=MISSING_COORD_FILLER,
                            device=device,
                            dropout=args.dropout,
                            fraction_complete_tf=args.fraction_complete_tf,
                            fraction_subseq_tf=args.fraction_subseq_tf,
                            angle_mean_path=args.angle_mean_path)
    else:
        raise argparse.ArgumentError("Model architecture not implemented.")
    return model


def seed_rngs(args):
    """
    Seed all necessary random number generators.
    """
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    """
    Argument parsing, model loading, and model training.
    """
    global LOGFILEHEADER
    global START_EPOCH
    global START_TIME
    global MISSING_COORD_FILLER
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = 500
    START_EPOCH = 0
    START_TIME = time.time()
    MISSING_COORD_FILLER = 0

    torch.set_printoptions(precision=5, sci_mode=False)
    parser = argparse.ArgumentParser()

    # Required args
    required = parser.add_argument_group("Required Args")
    required.add_argument('data', help="Path to training data.")
    required.add_argument("name", type=str, help="The model name.")

    # Training parameters
    training = parser.add_argument_group("Training Args")
    training.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    training.add_argument('-e', '--epochs', type=int, default=10)
    training.add_argument("-b", '--batch_size', type=int, default=8)
    training.add_argument('-es', '--early_stopping', type=int, default=None,
                        help="Stops if training hasn't improved in X epochs")
    training.add_argument('-nws', '--n_warmup_steps', type=int, default=10_000,
                        help="Number of warmup training steps when using lr-scheduling as proposed in the original"
                             "Transformer paper.")
    training.add_argument('-cg', '--clip', type=float, default=1, help="Gradient clipping value.")
    training.add_argument('-l', '--loss', choices=["mse", "drmsd", "ln-drmsd", "combined"], default="combined",
                        help="Loss used to train the model. Can be root mean squared error (RMSE), distance-based root mean squared distance (DRMSD), length-normalized DRMSD (ln-DRMSD) or a combinaation of RMSE and ln-DRMSD.")
    training.add_argument('--train_only', action='store_true',
                        help="Train, validation, and testing sets are the same. Only report train accuracy.")
    training.add_argument('--lr_scheduling', action='store_true',
                        help='Use learning rate scheduling as described in original paper.')
    training.add_argument('--without_angle_means', action='store_true',
                        help="Do not initialize the model with pre-computed angle means.")
    training.add_argument('--eval_train', action='store_true',
                        help="Perform an evaluation of the entire training set after a training epoch.")
    training.add_argument('-opt', '--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help="Training optimizer.")
    training.add_argument("-fctf", "--fraction_complete_tf", type=float, default=1,
                        help="Fraction of the time to use teacher forcing for every timestep of the batch. Model trains"
                             "fastest when this is 1.")
    training.add_argument("-fsstf", "--fraction_subseq_tf", type=float, default=1,
                        help="Fraction of the time to use teacher forcing on a per-timestep basis.")
    training.add_argument("--skip_missing_res_train", action="store_true",
                        help="When training, skip over batches that have missing residues. This can make training"
                             "faster if using teacher forcing.")
    training.add_argument("--repeat_train", type=int, default=1,
                        help="Duplicate the training set X times. Useful for training on small datasets.")
    training.add_argument("-s", "--seed", type=float, default=11_731,
                          help="The random number generator seed for numpy "
                               "and torch.")

    # Model parameters
    model_args = parser.add_argument_group("Model Args")
    model_args.add_argument('-m', '--model', type=str, choices=["enc-dec", "enc-only"], default="enc-only",
                        help="Model architecture type. Encoder only or encoder/decoder model.")
    model_args.add_argument('-dm', '--d_model', type=int, default=512,
                        help="Dimension of each sequence item in the model. Each layer uses the same dimension for "
                             "simplicity.")
    model_args.add_argument('-dih', '--d_inner_hid', type=int, default=2048,
                        help="Dimmension of the inner layer of the feed-forward layer at the end of every Transformer"
                             " block.")
    model_args.add_argument('-nh', '--n_head', type=int, default=8, help="Number of attention heads.")
    model_args.add_argument('-nl', '--n_layers', type=int, default=6,
                        help="Number of layers in the model. If using encoder/decoder model, the encoder and decoder"
                             " both have this number of layers.")
    model_args.add_argument('-do', '--dropout', type=float, default=0.1, help="Dropout applied between layers.")
    model_args.add_argument('--postnorm', action='store_true',
                        help="Use post-layer normalization, as depicted in the original figure for the Transformer "
                             "model. May not train as well as pre-layer normalization.")
    model_args.add_argument("--angle_mean_path", type=str, default="protein/casp12_190927_100_angle_means.npy",
                        help="Path to vector of means for every predicted angle. Used to initialize model output.")


    # Saving args
    saving_args = parser.add_argument_group("Saving Args")
    saving_args.add_argument('--log_structure_step', type=int, default=10,
                        help="Frequency of logging structure data during training.")
    saving_args.add_argument('--log_wandb_step', type=int, default=1,
                        help="Frequency of logging to wandb during training.")
    saving_args.add_argument('--no_cuda', action='store_true')
    saving_args.add_argument('--cluster', action='store_true', help="Set of parameters to facilitate training on a remote" +
                                                               " cluster. Limited I/O, etc.")
    saving_args.add_argument('--restart', action='store_true', help="Does not resume training.")
    saving_args.add_argument('--restart_opt', action='store_true', help="Resumes training but does not load the optimizer"
                                                                   "state. ")

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.buffering_mode = 1
    LOGFILEHEADER = prepare_log_header(args)
    seed_rngs(argparse)

    # Load dataset
    args.add_sos_eos = args.model == "enc-dec"
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings']["max_len"]
    training_data, validation_data, test_data = prepare_dataloaders(data, args, MAX_SEQ_LEN)

    # Prepare model
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = make_model(args, device).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                               betas=(0.9, 0.98), eps=1e-09, lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                              lr=args.learning_rate)
    if args.lr_scheduling:
        optimizer = ScheduledOptim(optimizer, args.d_model, args.n_warmup_steps)

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

    # Prepare Weights and Biases logging
    wandb.init(project="protein-transformer", entity="koes-group")
    wandb.watch(model, "all")
    wandb.config.update(args)
    if type(data["date"]) == set:
        wandb.config.update({"data_creation_date": next(iter(data["date"]))})
    else:
        wandb.config.update({"data_creation_date": data["date"]})
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"n_params": n_params,
                         "n_trainable_params": n_trainable_params,
                         "max_seq_len": MAX_SEQ_LEN})

    print(args, "\n")

    # Begin training
    train(model, metrics, training_data, validation_data, test_data, optimizer,
          device, args, log_writer)
    log_f.close()


if __name__ == '__main__':
    main()
