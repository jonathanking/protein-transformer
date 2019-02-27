'''
This script handling the training process.
'''

import argparse
import os
import time

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data
from dataset import paired_collate_fn, ProteinDataset
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from losses import cal_loss


def train_epoch(model, training_data, optimizer, device, opt, log_train_file):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_batches = 0.0
    loss = None
    training_losses = []
    if not opt.print_loss:
        pbar = tqdm(training_data, mininterval=2, desc='  - (Training) Loss = {0}   '.format(loss), leave=False)
    else:
        pbar = training_data

    for batch_num, batch in enumerate(pbar):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, loss_norm = cal_loss(pred, gold, src_seq, device, combined=opt.combined_loss)
        training_losses.append(float(loss_norm))
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

        if opt.print_loss and len(training_losses) > 32:
            print('Loss = {0:.6f}, NLoss = {3:.2f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses[-32:]), optimizer.cur_lr, loss_norm))
        elif opt.print_loss and len(training_losses) <= 32:
            print('Loss = {0:.6f}, NLoss = {3:.2f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses), optimizer.cur_lr, loss_norm))
        # elif opt.print_loss:
        #     print('Loss = {0:.6f}, NLoss = {2:.2f}, LR = {1:.7f}'.format(float(loss), optimizer.cur_lr, float(loss_norm)))

        # update parameters
        optimizer.step_and_update_lr()
        # optimizer.step()

        # note keeping
        total_loss += loss.item()
        n_batches += 1

        if not opt.print_loss and len(training_losses) > 32:
            pbar.set_description('  - (Training) Loss = {0:.6f}, NLoss = {3:.2f}, 32avg = {1:.6f}, LR = {2:.7f}'.format(
                float(loss), np.mean(training_losses[-32:]), optimizer.cur_lr, loss_norm))
        elif not opt.print_loss:
            pbar.set_description('  - (Training) Loss = {0:.6f}, NLoss = {2:.2f}, LR = {1:.7f}'.format(float(loss), optimizer.cur_lr, loss_norm))

        if opt.batch_log and  batch_num % 100 == 0:
            save_model_and_log_per_batch(opt, model, loss_norm, training_losses, log_train_file)


    return total_loss / n_batches

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_batches = 0.0


    with torch.no_grad():
        for batch in validation_data:
            # tqdm(
            #     validation_data, mininterval=2,
            #     desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)

            gold = tgt_seq[:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, loss_norm = cal_loss(pred, gold, src_seq, device)

            # note keeping
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches

def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    # Set up training/validation log files.
    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'
        os.makedirs(os.path.dirname(log_train_file), exist_ok=True)

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss\n')
            log_vf.write('epoch,loss\n')

    valid_losses = []
    epoch_last_improved = -1
    best_valid_loss_so_far = 10000
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch(model, training_data, optimizer, device, opt, log_train_file)
        print('  - (Training)   loss: {loss: 8.5f} '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss = eval_epoch(model, validation_data, device)
        print('  - (Validation) loss: {loss: 8.5f}, '\
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss,
                    elapse=(time.time()-start)/60))

        valid_losses.append(valid_loss)

        if opt.step_when and valid_loss < best_valid_loss_so_far:
            best_valid_loss_so_far = valid_loss
            epoch_last_improved = epoch_i
        elif opt.step_when and epoch_i - epoch_last_improved > opt.step_when:
            # Model hasn't improved in 100 epochs
            print("No improvement for 100 epochs. Stopping model training early.")
            break



        save_model_and_log(opt, model, valid_loss, valid_losses, log_train_file, log_valid_file, train_loss, epoch_i)




def save_model_and_log(opt, model, valid_loss, valid_losses, log_train_file, log_valid_file, train_loss, epoch_i):
    # Record model state and log training info
    model_state_dict = model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'settings': opt,
        'epoch': epoch_i}

    if opt.save_model:
        os.makedirs(os.path.dirname(opt.save_model), exist_ok=True)
        if opt.save_mode == 'all':
            model_name = opt.save_model + '_loss_{vloss:3.3f}.chkpt'.format(vloss=valid_loss)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = opt.save_model + '.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')
    if log_train_file and log_valid_file:
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            if not opt.batch_log:
                log_tf.write('{epoch},{loss: 8.5f}\n'.format(
                    epoch=epoch_i, loss=train_loss))
            log_vf.write('{epoch},{loss: 8.5f}\n'.format(
                epoch=epoch_i, loss=valid_loss))

def save_model_and_log_per_batch(opt, model, train_loss, train_losses, log_train_file):
    # Record model state and log training info
    model_state_dict = model.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'settings': opt}

    if opt.save_model:
        model_name = opt.save_model + '_loss_{vloss:3.3f}.chkpt'.format(vloss=train_loss)
        torch.save(checkpoint, model_name)
        print('    - [Info] The checkpoint file has been updated.')
    if log_train_file:
        with open(log_train_file, 'a') as log_tf:
            log_tf.write('{loss: 8.5f}\n'.format(loss=train_loss))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-step_when', type=int, default=None)
    parser.add_argument('-clip', type=float, default=1.0)
    parser.add_argument('-combined_loss', action='store_true', help="Use a loss that combines (quasi-equally) DRMSD and MSE.")

    parser.add_argument('-d_word_vec', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=75)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-batch_log', action='store_true', help="Save the model on a batch performance basis. Uses length-normalized DRMSD.")
    parser.add_argument('-print_loss', action='store_true')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings']["max_len"]

    training_data, validation_data = prepare_dataloaders(data, opt)

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
            betas=(0.9, 0.98), eps=1e-09, lr=1e-5),
        opt.d_model, opt.n_warmup_steps)

    # optimizer =  optim.Adam(filter(lambda x: x.requires_grad, transformer.parameters()),
    #                         betas=(0.9, 0.98), eps=1e-09, lr=1e-3)

    train(transformer, training_data, validation_data, optimizer, device ,opt)

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
