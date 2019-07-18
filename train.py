import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from dataset import paired_collate_fn, ProteinDataset
from losses import drmsd_loss, mse_loss, combine_drmsd_mse
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

LOGFILEHEADER = ''


def train_epoch(model, training_data, optimizer, device, opt, log_writer):
    """ Epoch operation in training phase"""
    model.train()

    total_drmsd_loss = 0
    total_mse_loss = 0
    n_batches = 0.0
    loss = ""
    training_losses = []
    if not opt.cluster:
        pbar = tqdm(training_data, mininterval=2, desc='  - (Train) Loss = {0}   '.format(loss), leave=False)
    else:
        pbar = training_data

    for batch_num, batch in enumerate(pbar):
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:]

        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
        d_loss = drmsd_loss(pred, gold, src_seq, device)
        m_loss = mse_loss(pred, gold)
        c_loss = combine_drmsd_mse(d_loss, m_loss, w=0.5)
        if opt.combined_loss:
            loss = c_loss
        else:
            loss = d_loss
        loss.backward()
        training_losses.append(float(loss))

        # Clip gradients
        if opt.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

        # update parameters
        optimizer.step()
        if opt.lr_scheduling:
            cur_lr = optimizer.cur_lr
            lr_string = f"{cur_lr:.7f}"
        else:
            cur_lr = 0
            lr_string = ""

        # note keeping
        total_drmsd_loss += d_loss.item()
        total_mse_loss += m_loss.item()
        n_batches += 1
        if not opt.cluster and len(training_losses) > 32:
            pbar.set_description('  - (Train) drmsd = {0:.6f}, rmse = {3:.6f}, 32avg = {1:.6f}, comb = {4:.6f}, '
                                 'LR = {2}'.format(float(d_loss), np.mean(training_losses[-32:]), lr_string,
                                                       np.sqrt(float(m_loss)), float(c_loss)))
        elif not opt.cluster:
            pbar.set_description('  - (Train) drmsd = {0:.6f}, rmse = {2:.6f}, comb = {3:.6f}, LR = {1}'.format(
                float(d_loss), lr_string, np.sqrt(float(m_loss)), float(c_loss)))
        if opt.cluster and len(training_losses) > 32:
            print('Loss = {0:.6f}, 32avg = {1:.6f}, LR = {2}'.format(
                float(loss), np.mean(training_losses[-32:]), lr_string))
        elif opt.cluster and len(training_losses) <= 32:
            print('Loss = {0:.6f}, 32avg = {1:.6f}, LR = {2}'.format(
                float(loss), np.mean(training_losses), lr_string))

        log_batch(log_writer, d_loss.item(), m_loss.item(), None, c_loss.item(), cur_lr, is_val=False,
                  end_of_epoch=False, t=time.time())

        if np.isnan(loss.item()):
            print("A nan loss has occurred. Exiting training.")
            sys.exit(1)

    return total_drmsd_loss / n_batches, total_mse_loss / n_batches


def eval_epoch(model, validation_data, device, opt, mode="Val"):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_drmsd_loss = 0
    total_mse_loss = 0
    total_rmsd_loss = 0
    total_combined_loss = 0
    n_batches = 0.0
    loss = ""
    if not opt.cluster:
        pbar = tqdm(validation_data, mininterval=2, desc='  - ({0}) Loss = {1}   '.format(mode, loss), leave=False)
    else:
        pbar = validation_data

    with torch.no_grad():
        for batch in pbar:
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:]
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            d_loss, r_loss = drmsd_loss(pred, gold, src_seq, device,
                                        return_rmsd=True)  # When evaluating the epoch, only use DRMSD loss
            m_loss = mse_loss(pred, gold)
            c_loss = combine_drmsd_mse(d_loss, m_loss)
            total_drmsd_loss += d_loss.item()
            total_mse_loss += m_loss.item()
            total_rmsd_loss += r_loss
            total_combined_loss += c_loss.item()
            n_batches += 1

            if not opt.cluster:
                pbar.set_description('  - ({1}) drmsd = {0:.6f}, rmse = {2:.6f}, comb = {3:.6f}'.format(
                    float(d_loss), mode, np.sqrt(float(m_loss)), float(c_loss)))

    return (x / n_batches for x in [total_drmsd_loss, total_mse_loss, total_rmsd_loss, total_combined_loss])


def train(model, training_data, validation_data, test_data, optimizer, device, opt, log_writer):
    """ Start training. """

    valid_drmsd_losses = []
    valid_combined_losses = []
    train_combined_losses = []
    train_drmsd_losses = []
    epoch_last_improved = -1
    best_valid_loss_so_far = np.inf
    for epoch_i in range(opt.epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        _, _ = train_epoch(model, training_data, optimizer, device, opt, log_writer)
        train_drmsd_loss, train_mse_loss, train_rmsd_loss, train_comb_loss = eval_epoch(model, training_data, device,
                                                                                        opt, mode="Train")
        train_combined_losses.append(train_comb_loss)
        train_drmsd_losses.append(train_drmsd_loss)
        if opt.lr_scheduling:
            cur_lr = optimizer.cur_lr
        else:
            cur_lr = 0
        print('  - (Train)   drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min, lr: {lr: {lr_precision}} '.format(d=train_drmsd_loss,
                                                                            m=np.sqrt(train_mse_loss),
                                                                            elapse=(time.time() - start) / 60,
                                                                            lr=cur_lr, rmsd=train_rmsd_loss,
                                                                            comb=train_comb_loss,
                                                                            lr_precision="5.2e"
                                                                            if (cur_lr < .001 and cur_lr != 0) else
                                                                            "5.3f"))

        if not opt.train_only:
            start = time.time()
            val_drmsd_loss, val_mse_loss, val_rmsd_loss, val_comb_loss = eval_epoch(model, validation_data, device, opt)
            print('  - (Validation) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
                  'elapse: {elapse:3.3f} min'.format(d=val_drmsd_loss, m=np.sqrt(val_mse_loss),
                                                     elapse=(time.time() - start) / 60, rmsd=val_rmsd_loss,
                                                     comb=val_comb_loss))
            log_batch(log_writer, val_drmsd_loss, val_mse_loss, val_rmsd_loss, val_comb_loss, cur_lr,
                      is_val=True, end_of_epoch=True)
            valid_drmsd_losses.append(val_drmsd_loss)
            valid_combined_losses.append(val_comb_loss)

        log_batch(log_writer, train_drmsd_loss, train_mse_loss, train_rmsd_loss, train_comb_loss, cur_lr,
                  is_val=False, end_of_epoch=True)

        if opt.combined_loss and not opt.train_only:
            loss_to_compare = val_comb_loss
            losses_to_compare = valid_combined_losses
        elif opt.combined_loss and opt.train_only:
            loss_to_compare = train_comb_loss
            losses_to_compare = train_combined_losses
        elif not opt.combined_loss and not opt.train_only:
            loss_to_compare = val_drmsd_loss
            losses_to_compare = valid_drmsd_losses
        elif not opt.combined_loss and opt.train_only:
            loss_to_compare = train_drmsd_loss
            losses_to_compare = train_drmsd_losses

        if opt.early_stopping and loss_to_compare < best_valid_loss_so_far:
            best_valid_loss_so_far = loss_to_compare
            epoch_last_improved = epoch_i
        elif opt.early_stopping and epoch_i - epoch_last_improved > opt.early_stopping:
            # Model hasn't improved in X epochs
            print("No improvement for {} epochs. Stopping model training early.".format(opt.early_stopping))
            break
        save_model(opt, optimizer, model, loss_to_compare, losses_to_compare, epoch_i)

    if not opt.train_only:
        # Evaluate model on test set
        t = time.time()
        test_drmsd_loss, test_mse_loss, test_rmsd_loss, test_comb_loss = eval_epoch(model, test_data, device, opt)
        print('  - (Test) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min'.format(d=test_drmsd_loss, m=np.sqrt(test_mse_loss),
                                                 elapse=(time.time() - t) / 60, comb=test_comb_loss,
                                                 rmsd=test_rmsd_loss))
        log_batch(log_writer, test_drmsd_loss, test_mse_loss, test_rmsd_loss, test_comb_loss, cur_lr,
                  is_val=True,
                  end_of_epoch=True)


def save_model(opt, optimizer, model, valid_loss, valid_losses, epoch_i):
    """ Records model state according to a checkpointing policy. Defaults to best validation set performance. """
    model_state_dict = model.state_dict()
    checkpoint = {
        'model_state_dict': model_state_dict,
        'settings': opt,
        'epoch': epoch_i,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_loss}

    if opt.save_mode == 'all':
        chkpt_file_name = opt.chkpt_path + "_epoch-{0}_vloss-{1}.chkpt".format(epoch_i, valid_loss)
        torch.save(checkpoint, chkpt_file_name)
    if len(valid_losses) == 1 or valid_loss < min(valid_losses[:-1]):
        chkpt_file_name = opt.chkpt_path + "_best.chkpt".format(epoch_i, valid_loss)
        torch.save(checkpoint, chkpt_file_name)
        print('    - [Info] The checkpoint file has been updated.')


def log_batch(log_writer, drmsd, mse, rmsd, combined, cur_lr, is_val=False, end_of_epoch=False, t=time.time()):
    """ Logs training info to a predetermined log. """
    log_writer.writerow([drmsd, np.sqrt(mse), rmsd, combined, cur_lr, is_val, end_of_epoch, t])


def prepare_log_header(opt):
    """ Returns the column ordering for the logfile. """
    if opt.combined_loss:
        return 'drmsd,rmse,rmsd,combined,lr,is_val,is_end_of_epoch,time\n'
    else:
        return 'drmsd,rmse,rmsd,lr,is_val,is_end_of_epoch,time\n'


def main():
    """ Main function """
    global LOGFILEHEADER
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
    parser.add_argument('-nws', '--n_warmup_steps', type=int, default=1000)
    parser.add_argument('-cg', '--clip', type=float, default=None)
    parser.add_argument('-cl', '--combined_loss', action='store_true',
                        help="Use a loss that combines (quasi-equally) DRMSD and MSE.")
    parser.add_argument('--train_only', action='store_true',
                        help="Train, validation, and testing sets are the same. Only report train accuracy.")
    parser.add_argument('--lr_scheduling', action='store_true', help='Use learning rate scheduling as described in" + '
                                                                     '"original paper.')
    parser.add_argument('--without_angle_means', action='store_true',
                        help="Do not initialize the model with pre-computed angle means.")

    # Model parameters
    parser.add_argument('-dwv', '--d_word_vec', type=int, default=20)
    parser.add_argument('-dm', '--d_model', type=int, default=512)
    parser.add_argument('-dih', '--d_inner_hid', type=int, default=2048)
    parser.add_argument('-dk', '--d_k', type=int, default=64)
    parser.add_argument('-dv', '--d_v', type=int, default=64)
    parser.add_argument('-nh', '--n_head', type=int, default=8)
    parser.add_argument('-nl', '--n_layers', type=int, default=6)
    parser.add_argument('-do', '--dropout', type=float, default=0)

    # Saving args
    parser.add_argument('--log', default=None, nargs=1)
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--cluster', action='store_true', help="Set of parameters to facilitate training on a remote" +
                                                               " cluster. Limited I/O, etc.")

    # Temporary args
    parser.add_argument('--proteinnet', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model
    args.buffering_mode = None if args.cluster else 1
    LOGFILEHEADER = prepare_log_header(args)

    # ========= Loading Dataset ========= #
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings']["max_len"]

    training_data, validation_data, test_data = prepare_dataloaders(data, args)

    # ========= Preparing Log and Checkpoint Files ========= #
    
    if not args.log:
        args.log_file = "./data/logs/" + args.name + '.train'
    else:
        args.log_file = "./data/logs/" + args.log + '.train'
    print(args, "\n")
    print('[Info] Training performance will be written to file: {}'.format(args.log_file))
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_f = open(args.log_file, 'w', buffering=args.buffering_mode)
    log_f.write(LOGFILEHEADER)
    log_writer = csv.writer(log_f)
    args.chkpt_path = "./data/checkpoints/" + args.name
    os.makedirs("./data/checkpoints", exist_ok=True)

    # ========= Preparing Model ========= #

    device = torch.device('cuda' if args.cuda else 'cpu')
    transformer = Transformer(args,
                              d_k=args.d_k,
                              d_v=args.d_v,
                              d_model=args.d_model,
                              d_inner=args.d_inner_hid,
                              n_layers=args.n_layers,
                              n_head=args.n_head,
                              dropout=args.dropout).to(device)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, transformer.parameters()),
                           betas=(0.9, 0.98), eps=1e-09, lr=args.learning_rate)
    if args.lr_scheduling:
        optimizer = ScheduledOptim(optimizer, args.d_model, args.n_warmup_steps, simple=False)

    train(transformer, training_data, validation_data, test_data, optimizer, device, args, log_writer)
    log_f.close()
    # TODO: Add test data evaluation


def prepare_dataloaders(data, opt):
    """ data is a dictionary containing all necessary training data."""
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['train']['seq'],
            angs=data['train']['ang'],
            ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    # TODO: load one or multiple validation sets
    if opt.proteinnet:
        valid_loader = torch.utils.data.DataLoader(
            ProteinDataset(
                seqs=data['valid'][70]['seq'],
                angs=data['valid'][70]['ang'],
                ),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)
    else:
        valid_loader = torch.utils.data.DataLoader(
            ProteinDataset(
                seqs=data['valid']['seq'],
                angs=data['valid']['ang'],
                ),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)

    test_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['test']['seq'],
            angs=data['test']['ang'],
            ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    main()
