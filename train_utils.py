""" A script to hold some utility functions for model training. """
import numpy as np
import time

def print_status(mode, opt, items):
    """
    Handles all status printing updates for the model. Allows complex string formatting per method while shrinking
    the number of lines of code per each training subroutine. mode is one of 'train_epoch', 'eval_epoch', 'train_train',
    'train_val', or 'train_test'.
    'train_epoch' refers to print statements within a training epoch,
    'eval_epoch' refers to print statemetns within an evaluation epoch,
    'train_train' refers to the end of a training epoch,
    'train_val' refers to the end of a validation epoch, and
    'train_test' refers to the end of a test epoch.
    """
    if mode == "train_epoch":
        pbar, metrics = items
        cur_lr = metrics["history-lr"][-1]
        training_losses = metrics["train"]["batch-history"]
        train_drmsd_loss = metrics["train"]["batch-drmsd"]
        train_mse_loss = metrics["train"]["batch-mse"]
        train_comb_loss = metrics["train"]["batch-combined"]
        if opt.combined_loss:
            loss = train_comb_loss
        else:
            loss = metrics["train"]["batch-ln-drmsd"]
        lr_string = f", LR = {cur_lr:.7f}" if opt.lr_scheduling else ""

        if not opt.cluster and len(training_losses) > 32:
            pbar.set_description('\r  - (Train) drmsd = {0:.6f}, ln-drmsd = {lnd:0.6f}, rmse = {3:.6f}, 32avg = {1:.6f}'
                                 ', comb = {4:.6f}{2}'.format(float(train_drmsd_loss), np.mean(training_losses[-32:]),
                                                              lr_string, np.sqrt(float(train_mse_loss)),
                                                              float(train_comb_loss),
                                                              lnd=metrics["train"]["batch-ln-drmsd"]))
        elif not opt.cluster:
            pbar.set_description('\r  - (Train) drmsd = {0:.6f}, ln-drmsd = {lnd:0.6f}, rmse = {2:.6f}, comb = '
                                 '{3:.6f}{1}'.format(float(train_drmsd_loss), lr_string, np.sqrt(float(train_mse_loss)),
                                                     float(train_comb_loss), lnd=metrics["train"]["batch-ln-drmsd"]))
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
        start, metrics = items
        cur_lr = metrics["history-lr"][-1]
        train_drmsd_loss = metrics["train"]["batch-drmsd"]
        train_mse_loss = metrics["train"]["batch-mse"]
        train_rmsd_loss_str = "{:6.3f}".format(metrics["train"]["batch-rmsd"]) if metrics["train"]["batch-rmsd"] else "nan"
        train_comb_loss = metrics["train"]["batch-combined"]
        print('\r  - (Train)   drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min, lr: {lr: {lr_precision}} '.format(d=train_drmsd_loss,
                                                                            m=np.sqrt(train_mse_loss),
                                                                            elapse=(time.time() - start) / 60,
                                                                            lr=cur_lr, rmsd=train_rmsd_loss_str,
                                                                            comb=train_comb_loss,
                                                                            lr_precision="5.2e"
                                                                            if (cur_lr < .001 and cur_lr != 0) else
                                                                            "5.3f"))
    elif mode == "train_val":
        start, metrics = items
        val_drmsd_loss = metrics["valid"]["epoch-drmsd"]
        val_mse_loss = metrics["valid"]["epoch-mse"]
        val_rmsd_loss = metrics["valid"]["epoch-rmsd"]
        val_comb_loss = metrics["valid"]["epoch-combined"]
        print('\r  - (Validation) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min'.format(d=val_drmsd_loss, m=np.sqrt(val_mse_loss),
                                                 elapse=(time.time() - start) / 60, rmsd=val_rmsd_loss,
                                                 comb=val_comb_loss))
    elif mode == "train_test":
        start, metrics = items
        test_drmsd_loss = metrics["test"]["epoch-drmsd"]
        test_mse_loss = metrics["test"]["epoch-mse"]
        test_rmsd_loss = metrics["test"]["epoch-rmsd"]
        test_comb_loss = metrics["test"]["epoch-combined"]
        print('\r  - (Test) drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd: 6.3f}, comb: {comb: 6.3f}, '
              'elapse: {elapse:3.3f} min'.format(d=test_drmsd_loss, m=np.sqrt(test_mse_loss),
                                                 elapse=(time.time() - start) / 60, comb=test_comb_loss,
                                                 rmsd=test_rmsd_loss))


def update_loss_trackers(opt, epoch_i, metrics):
    """ Updates the current loss to compare according to an early stopping policy."""

    loss_to_compare, losses_to_compare = None, None
    if opt.combined_loss and not opt.train_only:
        loss_to_compare = metrics["valid"]["epoch-combined"]
        losses_to_compare = metrics["valid"]["epoch-history-combined"]
    elif opt.combined_loss and opt.train_only:
        loss_to_compare = metrics["train"]["epoch-combined"]
        losses_to_compare = metrics["train"]["epoch-history-combined"]
    elif not opt.combined_loss and not opt.train_only:
        loss_to_compare = metrics["valid"]["epoch-drmsd"]
        losses_to_compare = metrics["valid"]["epoch-history-drmsd"]
    elif not opt.combined_loss and opt.train_only:
        loss_to_compare = metrics["train"]["epoch-drmsd"]
        losses_to_compare = metrics["train"]["epoch-history-drmsd"]

    if loss_to_compare < metrics["best_valid_loss_so_far"]:
        metrics["best_valid_loss_so_far"] = loss_to_compare
        metrics["epoch_last_improved"] = epoch_i
    elif opt.early_stopping and epoch_i - metrics["epoch_last_improved"] > opt.early_stopping:
        # Model hasn't improved in X epochs
        print("No improvement for {} epochs. Stopping model training early.".format(opt.early_stopping))
        raise EarlyStoppingCondition

    metrics["loss_to_compare"] = loss_to_compare
    metrics["losses_to_compare"] = losses_to_compare

    return metrics


def init_metrics(opt):
    """ Returns an empty metric dictionary for recording model performance. """
    # TODO add metrics class for tracking metrics, or use ignite
    metrics = {"train": {"epoch-history-drmsd": [],
                         "epoch-history-combined": []},
               "valid": {"epoch-history-drmsd": [],
                         "epoch-history-combined": []},
               "test":  {"epoch-history-drmsd": [],
                         "epoch-history-combined": []},
               "history-lr": [],
               "epoch_last_improved": -1,
               "best_valid_loss_so_far": np.inf
               }
    if not opt.lr_scheduling:
        metrics["history-lr"] = [0]
    return metrics


def update_metrics(metrics, mode, drmsd, ln_drmsd, mse, combined, rmsd=None, batch_level=True):
    """ Records relevant metrics in the metrics data structure while training.
        If batch_level is true, this means the loss for the current batch is
        recorded in addition to the running epoch loss.
    """
    if batch_level:
        metrics[mode]["batch-drmsd"] = drmsd.item()
        metrics[mode]["batch-ln-drmsd"] = ln_drmsd.item()
        metrics[mode]["batch-mse"] = mse.item()
        metrics[mode]["batch-combined"] = combined.item()
        if rmsd: metrics[mode]["batch-rmsd"] = rmsd.item()
    metrics[mode]["epoch-drmsd"] += drmsd.item()
    metrics[mode]["epoch-ln-drmsd"] += ln_drmsd.item()
    metrics[mode]["epoch-mse"] += mse.item()
    metrics[mode]["epoch-combined"] += combined.item()
    if rmsd: metrics[mode]["epoch-rmsd"] += rmsd.item()
    return metrics


def reset_metrics_for_epoch(metrics, mode):
    """ Resets the running and batch-specific metrics for a new epoch. """
    metrics[mode]["epoch-drmsd"] = metrics[mode]["batch-drmsd"] = 0
    metrics[mode]["epoch-ln-drmsd"] = metrics[mode]["batch-ln-drmsd"] = 0
    metrics[mode]["epoch-mse"] = metrics[mode]["batch-mse"] = 0
    metrics[mode]["epoch-combined"] = metrics[mode]["batch-combined"] = 0
    if mode == "train":
        metrics[mode]["epoch-rmsd"] = metrics[mode]["batch-rmsd"] = None
    else:
        metrics[mode]["epoch-rmsd"] = metrics[mode]["batch-rmsd"] = 0
    metrics[mode]["batch-history"] = []
    return metrics


def update_metrics_end_of_epoch(metrics, mode, n_batches):
    """ Averages the running metrics over an epoch """
    metrics[mode]["epoch-drmsd"] /= n_batches
    metrics[mode]["epoch-ln-drmsd"] /= n_batches
    metrics[mode]["epoch-mse"] /= n_batches
    metrics[mode]["epoch-combined"] /= n_batches
    # We don't bother to compute rmsd when training, but is included in the metrics for completeness
    if mode == "train":
        metrics[mode]["epoch-rmsd"] = None
    else:
        metrics[mode]["epoch-rmsd"] /= n_batches
    metrics[mode]["epoch-history-combined"].append(metrics[mode]["epoch-combined"])
    metrics[mode]["epoch-history-drmsd"].append(metrics[mode]["epoch-drmsd"])
    return metrics


def prepare_log_header(opt):
    """ Returns the column ordering for the logfile. """
    if opt.combined_loss:
        return 'drmsd,ln_drmsd,rmse,rmsd,combined,lr,mode,granularity,time\n'
    else:
        return 'drmsd,ln_drmsd,rmse,rmsd,lr,mode,granularity,time\n'


class EarlyStoppingCondition(Exception):
    """An exception to raise when Early Stopping conditions are met."""
    def __init__(self, *args):
        super().__init__(*args)
