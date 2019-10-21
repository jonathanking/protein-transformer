""" A script to hold some utility functions for model training. """
import numpy as np
import time

def print_status(mode, opt, items):
    """
    Handles all status printing updates for the model. Allows complex string formatting per method while shrinking
    the number of lines of code per each training subroutine. mode is one of 'train_epoch', 'eval_epoch', 'train_train',
    'train_val', or 'train_test'.
    """
    if mode == "train_epoch":
        pbar, loss, d_loss, d_loss_norm, training_losses, cur_lr, m_loss, c_loss = items
        lr_string = f", LR = {cur_lr:.7f}" if opt.lr_scheduling else ""
        if not opt.cluster and len(training_losses) > 32:
            pbar.set_description('\r  - (Train) drmsd = {0:.6f}, ln-drmsd = {lnd:0.6f}, rmse = {3:.6f}, 32avg = {1:.6f}'
                                 ', comb = {4:.6f}{2}'.format(float(d_loss), np.mean(training_losses[-32:]), lr_string,
                                                              np.sqrt(float(m_loss)), float(c_loss), lnd=d_loss_norm))
        elif not opt.cluster:
            pbar.set_description('\r  - (Train) drmsd = {0:.6f}, ln-drmsd = {lnd:0.6f}, rmse = {2:.6f}, comb = '
                                 '{3:.6f}{1}'.format(float(d_loss), lr_string, np.sqrt(float(m_loss)), float(c_loss),
                                                     lnd=d_loss_norm))
        if opt.cluster and len(training_losses) > 32:
            print('Loss = {0:.6f}, 32avg = {1:.6f}{2}'.format(
                float(loss), np.mean(training_losses[-32:]), lr_string))
        elif opt.cluster and len(training_losses) <= 32:
            print('Loss = {0:.6f}, 32avg = {1:.6f}{2}'.format(
                float(loss), np.mean(training_losses), lr_string))

    elif mode == "eval_epoch":
        pbar, d_loss, mode, m_loss, c_loss = items
        if not opt.cluster:
            pbar.set_description('\r  - (Eval-{1}) drmsd = {0:.6f}, rmse = {2:.6f}, comb = {3:.6f}'.format(
                float(d_loss), mode, np.sqrt(float(m_loss)), float(c_loss)))

    elif mode == "train_train":
        cur_lr, train_drmsd_loss, train_mse_loss, start, train_rmsd_loss, train_comb_loss = items
        print('\r  - (Train)   drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min, lr: {lr: {lr_precision}} '.format(d=train_drmsd_loss,
                                                                            m=np.sqrt(train_mse_loss),
                                                                            elapse=(time.time() - start) / 60,
                                                                            lr=cur_lr, rmsd=train_rmsd_loss,
                                                                            comb=train_comb_loss,
                                                                            lr_precision="5.2e"
                                                                            if (cur_lr < .001 and cur_lr != 0) else
                                                                            "5.3f"))
    elif mode == "train_val":
        val_drmsd_loss, val_mse_loss, start, val_rmsd_loss, val_comb_loss = items
        print('\r  - (Validation) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min'.format(d=val_drmsd_loss, m=np.sqrt(val_mse_loss),
                                                 elapse=(time.time() - start) / 60, rmsd=val_rmsd_loss,
                                                 comb=val_comb_loss))
    elif mode == "train_test":
        test_drmsd_loss, test_mse_loss, t, test_comb_loss, test_rmsd_loss = items
        print('\r  - (Test) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min'.format(d=test_drmsd_loss, m=np.sqrt(test_mse_loss),
                                                 elapse=(time.time() - t) / 60, comb=test_comb_loss,
                                                 rmsd=test_rmsd_loss))