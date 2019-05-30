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
    if not opt.print_loss:
        pbar = tqdm(training_data, mininterval=2, desc='  - (Training) Loss = {0}   '.format(loss), leave=False)
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
        training_losses.append(float(d_loss))

        # Clip gradients
        if opt.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

        if opt.print_loss and len(training_losses) > 32:
            print('Loss = {0:.6f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses[-32:]), optimizer.cur_lr))
        elif opt.print_loss and len(training_losses) <= 32:
            print('Loss = {0:.6f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses), optimizer.cur_lr))

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_drmsd_loss += d_loss.item()
        total_mse_loss += m_loss.item()
        n_batches += 1

        if not opt.print_loss and len(training_losses) > 32:
            pbar.set_description('  - (Training) drmsd = {0:.6f}, rmse = {3:.6f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(d_loss), np.mean(training_losses[-32:]), optimizer.cur_lr, np.sqrt(float(m_loss))))
        elif not opt.print_loss:
            pbar.set_description('  - (Training) drmsd = {0:.6f}, rmse = {2:.6f}, LR = {1:.7f}'.format(float(d_loss),
                                                                                                       optimizer.cur_lr,
                                                                                                       np.sqrt(float(
                                                                                                           m_loss))))

        log_batch(log_writer, d_loss.item(), m_loss.item(), None, c_loss, optimizer.cur_lr, is_val=False,
                  end_of_epoch=False, t=time.time())

        if np.isnan(loss.item()):
            print("A nan loss has occurred. Exiting training.")
            sys.exit(1)

    return total_drmsd_loss / n_batches, total_mse_loss / n_batches


def eval_epoch(model, validation_data, device, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_drmsd_loss = 0
    total_mse_loss = 0
    total_rmsd_loss = 0
    total_combined_loss = 0
    n_batches = 0.0

    with torch.no_grad():
        for batch in validation_data:
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

    return (x / n_batches for x in [total_drmsd_loss, total_mse_loss, total_rmsd_loss, total_combined_loss])


def train(model, training_data, validation_data, test_data, optimizer, device, opt, log_writer):
    """ Start training. """

    valid_drmsd_losses = []
    valid_combined_losses = []
    epoch_last_improved = -1
    best_valid_loss_so_far = np.inf
    for epoch_i in range(opt.epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_drmsd_loss, train_mse_loss = train_epoch(model, training_data, optimizer, device, opt, log_writer)
        train_drmsd_loss, train_mse_loss, train_rmsd_loss, train_comb_loss = eval_epoch(model, training_data, device,
                                                                                        opt)
        print('  - (Training)   drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min, lr: {lr: {lr_precision}} '.format(d=train_drmsd_loss,
                                                                            m=np.sqrt(train_mse_loss),
                                                                            elapse=(time.time() - start) / 60,
                                                                            lr=optimizer.cur_lr, rmsd=train_rmsd_loss,
                                                                            comb=train_comb_loss,
                                                                            lr_precision="5.2e"
                                                                            if optimizer.cur_lr < .001 else "5.3f"))

        start = time.time()
        val_drmsd_loss, val_mse_loss, val_rmsd_loss, val_comb_loss = eval_epoch(model, validation_data, device, opt)
        print('  - (Validation) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min'.format(d=val_drmsd_loss, m=np.sqrt(val_mse_loss),
                                                 elapse=(time.time() - start) / 60, rmsd=val_rmsd_loss,
                                                 comb=val_comb_loss))

        t = time.time()
        log_batch(log_writer, train_drmsd_loss, train_mse_loss, train_rmsd_loss, train_comb_loss, optimizer.cur_lr,
                  is_val=False, end_of_epoch=True, t=t)
        log_batch(log_writer, val_drmsd_loss, val_mse_loss, val_rmsd_loss, val_comb_loss, optimizer.cur_lr,
                  is_val=True, end_of_epoch=True, t=t)

        valid_drmsd_losses.append(val_drmsd_loss)
        valid_combined_losses.append(val_comb_loss)
        if opt.combined_loss:
            loss_to_compare = val_comb_loss
            losses_to_compare = valid_combined_losses
        else:
            loss_to_compare = val_drmsd_loss
            losses_to_compare = valid_drmsd_losses

        if opt.early_stopping and loss_to_compare < best_valid_loss_so_far:
            best_valid_loss_so_far = loss_to_compare
            epoch_last_improved = epoch_i
        elif opt.early_stopping and epoch_i - epoch_last_improved > opt.early_stopping:
            # Model hasn't improved in X epochs
            print("No improvement for {} epochs. Stopping model training early.".format(opt.early_stopping))
            break
        save_model(opt, optimizer, model, loss_to_compare, losses_to_compare, epoch_i)

    # Evaluate model on test set
    t = time.time()
    test_drmsd_loss, test_mse_loss, test_rmsd_loss, test_comb_loss = eval_epoch(model, test_data, device, opt)
    print('  - (Test) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
          'elapse: {elapse:3.3f} min'.format(d=test_drmsd_loss, m=np.sqrt(test_mse_loss),
                                             elapse=(time.time() - t) / 60, comb=test_comb_loss, rmsd=test_rmsd_loss))
    log_batch(log_writer, test_drmsd_loss, test_mse_loss, test_rmsd_loss, test_comb_loss, optimizer.cur_lr, is_val=True,
              end_of_epoch=True, t=t)


def save_model(opt, optimizer, model, valid_loss, valid_losses, epoch_i):
    """ Records model state according to a checkpointing policy. Defaults to best validation set performance. """
    model_state_dict = model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'settings': opt,
        'epoch': epoch_i,
        'optimizer': optimizer}

    if opt.save_mode == 'all':
        chkpt_file_name = opt.chkpt_path + "_epoch-{0}_vloss-{1}.chkpt".format(epoch_i, valid_loss)
        torch.save(checkpoint, chkpt_file_name)
    if len(valid_losses) == 1 or valid_loss < min(valid_losses[:-1]):
        chkpt_file_name = opt.chkpt_path + "_best.chkpt".format(epoch_i, valid_loss)
        torch.save(checkpoint, chkpt_file_name)
        print('    - [Info] The checkpoint file has been updated.')


def log_batch(log_writer, drmsd, mse, rmsd, combined, cur_lr, is_val=False, end_of_epoch=False, t=time.time()):
    """ Logs training info to a predetermined log. """
    log_writer.writerow([drmsd, mse, rmsd, combined, cur_lr, is_val, end_of_epoch, t])


def prepare_log_header(opt):
    """ Returns the column ordering for the logfile. """
    if opt.combined_loss:
        return 'drmsd,mse,rmsd,combined,lr,is_val,is_end_of_epoch,time\n'
    else:
        return 'drmsd,mse,rmsd,lr,is_val,is_end_of_epoch,time\n'


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
    parser.add_argument('--print_loss', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    LOGFILEHEADER = prepare_log_header(opt)

    # ========= Loading Dataset ========= #
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings']["max_len"]

    training_data, validation_data, test_data = prepare_dataloaders(data, opt)

    # ========= Preparing Log and Checkpoint Files ========= #
    if not opt.log:
        opt.log_file = "./logs/" + opt.name + '.train'
    else:
        opt.log_file = "./logs/" + opt.log + '.train'
    print(opt, "\n")
    print('[Info] Training performance will be written to file: {}'.format(opt.log_file))
    os.makedirs(os.path.dirname(opt.log_file), exist_ok=True)
    log_f = open(opt.log_file, 'w', buffering=1)
    log_f.write(LOGFILEHEADER)
    log_writer = csv.writer(log_f)
    opt.chkpt_path = "./checkpoints/" + opt.name
    os.makedirs("./checkpoints", exist_ok=True)

    # ========= Preparing Model ========= #

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.max_token_seq_len,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps, simple=False)

    train(transformer, training_data, validation_data, test_data, optimizer, device, opt, log_writer)
    log_f.close()
    # TODO: Add test data evaluation


def prepare_dataloaders(data, opt):
    """ data is a dictionary containing all necessary training data."""
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['train']['seq'],
            angs=data['train']['ang']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['valid']['seq'],
            angs=data['valid']['ang']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    test_loader = torch.utils.data.DataLoader(
        ProteinDataset(
            seqs=data['test']['seq'],
            angs=data['test']['ang']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    main()
