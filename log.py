""" A script to hold some utility functions for model logging. """
import numpy as np
import time
import wandb
import sys


def print_status(mode, args, items):
    """
    Handles all status printing updates for the model. Allows complex string formatting per method while shrinking
    the number of lines of code per each training subroutine.
    """
    if mode == "train_epoch":
        print_train_batch_status(args, items)
    elif mode == "eval_epoch":
        print_eval_batch_status(args, items)
    elif mode == "train_train":
        print_end_of_epoch_status("train", items)
    elif mode == "train_val":
        print_end_of_epoch_status("valid", items)
    elif mode == "train_test":
        print_end_of_epoch_status("test", items)


def print_train_batch_status(args, items):
    """
    Print the status line during training after a single batch update. Uses tqdm progress bar
    by default, unless the script is run in a high performance computing (cluster) env.
    """
    # Extract relevant metrics
    pbar, metrics, src_seq = items
    cur_lr = metrics["history-lr"][-1]
    training_losses = metrics["train"]["batch-history"]
    train_drmsd_loss = metrics["train"]["batch-drmsd"]
    train_mse_loss = metrics["train"]["batch-mse"]
    train_comb_loss = metrics["train"]["batch-combined"]
    if args.combined_loss:
        loss = train_comb_loss
    else:
        loss = metrics["train"]["batch-ln-drmsd"]
    lr_string = f", LR = {cur_lr:.7f}" if args.lr_scheduling else ""
    speed = metrics["train"]["speed"]
    if len(training_losses) <= 32:
        rolling_avg = np.mean(training_losses)
    else:
        rolling_avg = np.mean(training_losses[-32:])

    if args.cluster:
        print('Loss = {0:.6f}, 32avg = {1:.6f}{2}, speed = {speed}'.format(
            float(loss),
            rolling_avg,
            lr_string,
            speed=speed))
    else:
        pbar.set_description('\r  - (Train) drmsd = {0:.6f}, ln-drmsd = {lnd:0.6f}, rmse = {3:.6f},'
                             ' 32avg = {1:.6f}, comb = {4:.6f}{2}, res/sec = {speed}'.format(
            float(train_drmsd_loss),
            rolling_avg,
            lr_string,
            np.sqrt(float(train_mse_loss)),
            float(train_comb_loss),
            lnd=metrics["train"]["batch-ln-drmsd"],
            speed=speed))


def print_eval_batch_status(args, items):
    """
    Print the status line during evaluation after a single batch update.
    Will only be seen if using a progress bar. Otherwise, there is no information logged.
    """
    pbar, d_loss, mode, m_loss, c_loss = items
    if not args.cluster:
        pbar.set_description('\r  - (Eval-{1}) drmsd = {0:.6f}, rmse = {2:.6f}, comb = {3:.6f}'.format(
            float(d_loss),
            mode,
            np.sqrt(float(m_loss)),
            float(c_loss)))


def print_end_of_epoch_status(mode, items):
    """
    Prints the training status at the end of an epoch and updates wandb summary stats.
    """
    start, metrics = items
    cur_lr = metrics["history-lr"][-1]
    drmsd_loss = metrics[mode]["epoch-drmsd"]
    mse_loss = metrics[mode]["epoch-mse"]
    rmsd_loss_str = "{:6.3f}".format(metrics[mode]["epoch-rmsd"]) if metrics[mode]["epoch-rmsd"] else "nan"
    comb_loss = metrics[mode]["epoch-combined"]
    avg_speed = np.mean(metrics[mode]["speed-history"])
    print('\r  - ({mode})   drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd}, comb: {comb: 6.3f}, '
          'elapse: {elapse:3.3f} min, lr: {lr: {lr_precision}}, res/sec = {speed:.2f}'.format(
        mode=mode.capitalize(),
        d=drmsd_loss,
        m=np.sqrt(mse_loss),
        elapse=(time.time() - start) / 60,
        lr=cur_lr,
        rmsd=rmsd_loss_str,
        comb=comb_loss,
        lr_precision="5.2e" if (cur_lr < .001 and cur_lr != 0) else "5.3f",
        speed=avg_speed))
    # Log end of epoch stats with wandb
    wandb.run.summary[f"final_epoch_{mode}_drmsd"] = metrics[mode]["epoch-drmsd"]
    wandb.run.summary[f"final_epoch_{mode}_mse"] = metrics[mode]["epoch-mse"]
    wandb.run.summary[f"final_epoch_{mode}_rmsd"] = metrics[mode]["epoch-rmsd"]
    wandb.run.summary[f"final_epoch_{mode}_comb"] = metrics[mode]["epoch-combined"]
    wandb.run.summary[f"final_epoch_{mode}_speed"] = avg_speed


def update_loss_trackers(args, epoch_i, metrics):
    """
    Updates the current loss to compare according to an early stopping policy.
    """
    if args.train_only:
        mode = "train"
    else:
        mode = "valid"
    if args.combined_loss:
        loss_str = "combined"
    else:
        loss_str = "drmsd"

    loss_to_compare = metrics[mode][f"epoch-{loss_str}"]
    losses_to_compare = metrics[mode][f"epoch-history-{loss_str}"]

    if loss_to_compare < metrics["best_valid_loss_so_far"]:
        metrics["best_valid_loss_so_far"] = loss_to_compare
        metrics["epoch_last_improved"] = epoch_i
    elif args.early_stopping and epoch_i - metrics["epoch_last_improved"] > args.early_stopping:
        # Model hasn't improved in X epochs
        print("No improvement for {} epochs. Stopping model training early.".format(args.early_stopping))
        raise EarlyStoppingCondition

    metrics["loss_to_compare"] = loss_to_compare
    metrics["losses_to_compare"] = losses_to_compare

    return metrics


def log_batch(log_writer, metrics, start_time,  mode="valid", end_of_epoch=False, t=None):
    """
    Logs training info to an already instantiated CSV-writer log.
    """
    if not t:
        t = time.time()
    m = metrics[mode]
    if end_of_epoch:
        be = "epoch"
    else:
        be = "batch"
    log_writer.writerow([m[f"{be}-drmsd"], m[f"{be}-ln-drmsd"], np.sqrt(m[f"{be}-mse"]),
                         m[f"{be}-rmsd"], m[f"{be}-combined"], metrics["history-lr"][-1],
                         mode, "epoch", round(t - start_time, 4), m["speed"]])


def do_train_batch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, src_seq, loss, optimizer, args, log_writer, pbar, start_time):
    """
    Performs all necessary logging at the end of a batch in the training epoch.
    Updates custom metrics dictionary and wandb logs. Prints status of training.
    Also checks for NaN losses.
    """
    metrics = update_metrics(metrics, "train", d_loss, ln_d_loss, m_loss, c_loss, src_seq,
                             tracking_loss=loss, batch_level=True)
    log_batch(log_writer, metrics, start_time, mode="train", end_of_epoch=False)
    wandb.log({"Train RMSE": np.sqrt(m_loss.item()),
               "Train DRMSD": d_loss,
               "Train ln-DRMSD": ln_d_loss,
               "Train Combined Loss": c_loss,
               "Train Speed": metrics["train"]["speed"]}, commit=not args.lr_scheduling)
    if args.lr_scheduling:
        metrics["history-lr"].append(optimizer.cur_lr)
        wandb.log({"Learning Rate": optimizer.cur_lr})
    print_status("train_epoch", args, (pbar, metrics, src_seq))
    # Check for NaNs
    if np.isnan(loss.item()):
        print("A nan loss has occurred. Exiting training.")
        sys.exit(1)


def do_eval_epoch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, r_loss, src_seq, args, pbar, mode):
    """
    Performs all necessary logging at the end of an evaluation epoch.
    Updates custom metrics dictionary and wandb logs. Prints status of training.
    """
    metrics = update_metrics(metrics, mode, d_loss, ln_d_loss, m_loss, c_loss, src_seq, r_loss, batch_level=False)
    wandb.log({f"{mode.title()} RMSE": np.sqrt(m_loss.item()),
               f"{mode.title()} RMSD": r_loss,
               f"{mode.title()} DRMSD": d_loss,
               f"{mode.title()} ln-DRMSD": ln_d_loss,
               f"{mode.title()} Combined Loss": c_loss,
               f"{mode.title()} Speed": metrics[mode]["speed"]})
    print_status("eval_epoch", args, (pbar, d_loss, mode, m_loss, c_loss))



def init_metrics(args):
    """
    Returns an empty metric dictionary for recording model performance.
    """
    metrics = {"train": {"epoch-history-drmsd": [],
                         "epoch-history-combined": []},
               "valid": {"epoch-history-drmsd": [],
                         "epoch-history-combined": []},
               "test":  {"epoch-history-drmsd": [],
                         "epoch-history-combined": []},
               "history-lr": [],
               "epoch_last_improved": -1,
               "best_valid_loss_so_far": np.inf,
               }
    if not args.lr_scheduling:
        metrics["history-lr"] = [0]
    return metrics


def update_metrics(metrics, mode, drmsd, ln_drmsd, mse, combined, src_seq, rmsd=None, tracking_loss=None, batch_level=True):
    """
    Records relevant metrics in the metrics data structure while training.
    If batch_level is true, this means the loss for the current batch is
    recorded in addition to the running epoch loss.
    """
    # Update loss values
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

    # Compute and update speed
    num_res = (src_seq != 0).any(dim=-1).sum().item()
    metrics[mode]["speed"] = round(num_res / (time.time() - metrics[mode]["batch-time"]), 2)
    metrics[mode]["batch-time"] = time.time()
    metrics[mode]["speed-history"].append(metrics[mode]["speed"])

    if tracking_loss:
        metrics[mode]["batch-history"].append(float(tracking_loss))
    return metrics


def reset_metrics_for_epoch(metrics, mode):
    """
    Resets the running and batch-specific metrics for a new epoch.
    """
    metrics[mode]["epoch-drmsd"] = metrics[mode]["batch-drmsd"] = 0
    metrics[mode]["epoch-ln-drmsd"] = metrics[mode]["batch-ln-drmsd"] = 0
    metrics[mode]["epoch-mse"] = metrics[mode]["batch-mse"] = 0
    metrics[mode]["epoch-combined"] = metrics[mode]["batch-combined"] = 0
    if mode == "train":
        metrics[mode]["epoch-rmsd"] = metrics[mode]["batch-rmsd"] = None
    else:
        metrics[mode]["epoch-rmsd"] = metrics[mode]["batch-rmsd"] = 0
    metrics[mode]["batch-history"] = []
    metrics[mode]["batch-time"] = time.time()
    metrics[mode]["speed-history"] = []
    return metrics


def update_metrics_end_of_epoch(metrics, mode, n_batches):
    """
    Averages the running metrics over an epoch
    """
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


def prepare_log_header(args):
    """
    Returns the column ordering for the logfile.
    """
    if args.combined_loss:
        return 'drmsd,ln_drmsd,rmse,rmsd,combined,lr,mode,granularity,time,speed\n'
    else:
        return 'drmsd,ln_drmsd,rmse,rmsd,lr,mode,granularity,time,speed\n'


class EarlyStoppingCondition(Exception):
    """
    An exception to raise when Early Stopping conditions are met.
    """
    def __init__(self, *args):
        super().__init__(*args)
