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
from losses import cal_loss
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


def train_epoch(model, training_data, optimizer, device, opt, log_writer):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    total_nloss = 0
    n_batches = 0.0
    loss = None
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

        loss, loss_norm = cal_loss(pred, gold, src_seq, device, combined=opt.combined_loss)
        training_losses.append(float(loss_norm))
        loss.backward()

        # Clip gradients
        if opt.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

        if opt.print_loss and len(training_losses) > 32:
            print('Loss = {0:.6f}, NLoss = {3:.2f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses[-32:]), optimizer.cur_lr, loss_norm))
        elif opt.print_loss and len(training_losses) <= 32:
            print('Loss = {0:.6f}, NLoss = {3:.2f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses), optimizer.cur_lr, loss_norm))


        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()
        total_nloss += loss_norm.item()
        n_batches += 1

        if not opt.print_loss and len(training_losses) > 32:
            pbar.set_description('  - (Training) Loss = {0:.6f}, NLoss = {3:.2f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses[-32:]), optimizer.cur_lr, loss_norm))
        elif not opt.print_loss:
            pbar.set_description('  - (Training) Loss = {0:.6f}, NLoss = {2:.2f}, LR = {1:.7f}'.format(float(loss), optimizer.cur_lr, loss_norm))

        log_batch(log_writer, loss.item(), loss_norm.item(), is_val=False, is_end_of_epoch=False, time=time.time())

        if np.isnan(loss.item()):
            print("A nan loss has occurred. Exiting training.")
            sys.exit(1)

    return total_loss / n_batches, total_nloss / n_batches


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total_nloss = 0
    n_batches = 0.0

    with torch.no_grad():
        for batch in validation_data:
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:]
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, loss_norm = cal_loss(pred, gold, src_seq, device)
            total_loss += loss.item()
            total_nloss += loss_norm.item()
            n_batches += 1

    return total_loss / n_batches, total_nloss / n_batches


def train(model, training_data, validation_data, optimizer, device, opt, log_writer):
    ''' Start training '''

    valid_losses = []
    epoch_last_improved = -1
    best_valid_loss_so_far = np.inf
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_nloss = train_epoch(model, training_data, optimizer, device, opt, log_writer)
        print('  - (Training)   loss: {loss: 8.5f} '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_nloss = eval_epoch(model, validation_data, device)
        print('  - (Validation) loss: {loss: 8.5f}, '\
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss,
                    elapse=(time.time()-start)/60))

        t = time.time()
        log_batch(log_writer, train_loss, train_nloss, is_val=False, is_end_of_epoch=True, time=t)
        log_batch(log_writer, valid_loss, valid_nloss, is_val=True, is_end_of_epoch=True, time=t)

        valid_losses.append(valid_loss)

        if opt.early_stopping and valid_loss < best_valid_loss_so_far:
            best_valid_loss_so_far = valid_loss
            epoch_last_improved = epoch_i
        elif opt.early_stopping and epoch_i - epoch_last_improved > opt.early_stopping:
            # Model hasn't improved in X epochs
            print("No improvement for {} epochs. Stopping model training early.".format(opt.early_stopping))
            break

        save_model(opt, model, valid_loss, valid_losses, epoch_i)


def save_model(opt, model, valid_loss, valid_losses, epoch_i):
    # Record model state and log training info
    model_state_dict = model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'settings': opt,
        'epoch': epoch_i}

    if opt.save_mode == 'all':
        chkpt_file_name = opt.chkpt_path + "_epoch-{0}_vloss-{1}.chkpt".format(epoch_i, valid_loss)
        torch.save(checkpoint, chkpt_file_name)
    if valid_loss <= min(valid_losses):
        chkpt_file_name = opt.chkpt_path + "_best.chkpt".format(epoch_i, valid_loss)
        torch.save(checkpoint, chkpt_file_name)
        print('    - [Info] The checkpoint file has been updated.')


def log_batch(log_writer, loss, nloss, is_val=False, is_end_of_epoch=False, time=time.time()):
    # Record model state and log training info, 'loss,nloss,is_val,is_end_of_epoch,time\n'
    log_writer.writerow([loss, nloss, is_val, is_end_of_epoch, time])

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument('-data', required=True, help="Path to training data.")
    parser.add_argument("-name", type=str, required=True, help="The model name.")

    # Training parameters
    parser.add_argument("-lr", "--learning_rate", type=float, default=1 * (10 ** -6))
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument("-b", '--batch_size', type=int, default=8)
    parser.add_argument('-early_stopping', type=int, default=None)
    parser.add_argument('-n_warmup_steps', type=int, default=75)
    parser.add_argument('-clip', type=float, default=None)
    parser.add_argument('-combined_loss', action='store_true', help="Use a loss that combines (quasi-equally) DRMSD and MSE.")

    # Model parameters
    parser.add_argument('-d_word_vec', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0)

    # Saving args
    parser.add_argument('-log', default=None, nargs="?")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-print_loss', action='store_true')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings']["max_len"]

    training_data, validation_data = prepare_dataloaders(data, opt)

    # ========= Preparing Log and Checkpoint Files =========#
    if not opt.log:
        opt.log_file = "./logs/" + opt.name + '.train'
    else:
        opt.log_file = "./logs/" + opt.log + '.train'
    print('[Info] Training performance will be written to file: {}'.format(opt.log_file))
    os.makedirs(os.path.dirname(opt.log_file), exist_ok=True)
    log_f = open(opt.log_file, 'w', buffering=1)
    log_f.write('loss,nloss,is_val,is_end_of_epoch,time\n')
    log_writer = csv.writer(log_f)
    opt.chkpt_path = "./checkpoints/" + opt.name
    os.makedirs("./checkpoints", exist_ok=True)

    #========= Preparing Model =========#

    print(opt)
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
            betas=(0.9, 0.98), eps=1e-09, lr=opt.learning_rate),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt, log_writer)
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
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
