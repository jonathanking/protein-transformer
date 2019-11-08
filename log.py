""" A script to hold some utility functions for model logging. """
import numpy as np
import time
import wandb
import sys
import torch

from dataset import VOCAB
from protein.PDB_Creator import PDB_Creator
from losses import angles_to_coords, inverse_trig_transform

def print_train_batch_status(args, items):
    """
    Print the status line during training after a single batch update. Uses tqdm progress bar
    by default, unless the script is run in a high performance computing (cluster) env.
    """
    # Extract relevant metrics
    pbar, metrics, src_seq = items
    cur_lr = metrics["history-lr"][-1]
    train_drmsd_loss = metrics["train"]["batch-drmsd"]
    train_mse_loss = metrics["train"]["batch-mse"]
    train_comb_loss = metrics["train"]["batch-combined"]
    if args.loss  == "combined":
        loss = train_comb_loss
    else:
        loss = metrics["train"]["batch-ln-drmsd"]
    lr_string = f", LR = {cur_lr:.7f}" if args.lr_scheduling else ""
    speed_avg = np.mean(metrics["train"]["speeds"])

    if args.cluster:
        print('Loss = {0:.6f}{1}, res/s={speed:.0f}'.format(
            float(loss),
            lr_string,
            speed=speed_avg))
    else:
        pbar.set_description('\r  - (Train) drmsd={drmsd:.2f}, lndrmsd={lnd:0.7f}, rmse={rmse:.4f},'
                             ' c={comb:.2f}{lr}, res/s={speed:.0f}'.format(
            drmsd=float(train_drmsd_loss),
            lr=lr_string,
            rmse=np.sqrt(float(train_mse_loss)),
            comb=float(train_comb_loss),
            lnd=metrics["train"]["batch-ln-drmsd"],
            speed=speed_avg))


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
    rmsd_loss_str = "{:6.3f}".format(metrics[mode]["epoch-rmsd"]) if metrics[mode]["epoch-rmsd"] != 0 else "nan"
    comb_loss = metrics[mode]["epoch-combined"]
    avg_speed = np.mean(metrics[mode]["speed-history"])
    print('\r  - ({mode})   drmsd: {d: 6.3f}, rmse: {m: 6.3f}, rmsd: {rmsd}, comb: {comb: 6.3f}, '
          'elapse: {elapse:3.3f} min, lr: {lr: {lr_precision}}, res/sec = {speed:.0f}'.format(
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
    loss_str = args.loss

    loss_to_compare = metrics[mode][f"epoch-{loss_str}"]
    losses_to_compare = metrics[mode][f"epoch-history-{loss_str}"]

    if loss_to_compare < metrics["best_valid_loss_so_far"]:
        metrics["best_valid_loss_so_far"] = loss_to_compare
        metrics["epoch_last_improved"] = epoch_i
    elif args.early_stopping and epoch_i - metrics["epoch_last_improved"] > args.early_stopping:
        # Model hasn't improved in X epochs
        print("No improvement for {} epochs. Stopping model training early.".format(args.early_stopping))
        wandb.run.summary["stopped_training_early"] = True
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


def do_train_batch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, src_seq, loss, optimizer, args, log_writer, pbar,
                           start_time, pred_angs, tgt_coords, step):
    """
    Performs all necessary logging at the end of a batch in the training epoch.
    Updates custom metrics dictionary and wandb logs. Prints status of training.
    Also checks for NaN losses.
    """
    do_log_str = not step or step % args.log_structure_step == 0
    do_log_lr  = args.lr_scheduling and (not step or args.log_wandb_step % step == 0)

    metrics = update_metrics(metrics, "train", d_loss, ln_d_loss, m_loss, c_loss, src_seq,
                             tracking_loss=loss, batch_level=True)
    if not step or step % args.log_wandb_step == 0:
        wandb.log({"Train Batch RMSE": np.sqrt(m_loss.item()),
                   "Train Batch DRMSD": d_loss,
                   "Train Batch ln-DRMSD": ln_d_loss,
                   "Train Batch Combined Loss": c_loss,
                   "Train Batch Speed": metrics["train"]["speed"]}, commit=not do_log_lr and not do_log_str)
    if args.lr_scheduling:
        metrics["history-lr"].append(optimizer.cur_lr)
        if not step or step % args.log_wandb_step  == 0:
            wandb.log({"Learning Rate": optimizer.cur_lr}, commit=not do_log_str)
    log_batch(log_writer, metrics, start_time, mode="train", end_of_epoch=False)
    print_train_batch_status(args, (pbar, metrics, src_seq))
    # Check for NaNs
    if np.isnan(loss.item()):
        print("A nan loss has occurred. Exiting training.")
        sys.exit(1)
    if do_log_str:
        with torch.no_grad():
            pred_coords = angles_to_coords(inverse_trig_transform(pred_angs)[-1].cpu(), src_seq[-1].cpu(), remove_batch_padding=True)
        log_structure(args, pred_coords, tgt_coords, src_seq[-1])


def log_structure(args, pred_coords, gold_item, src_seq):
    """
    Logs a 3D structure prediction to wandb.
    # TODO save PDB files with numbers in addition to just GLTF files
    """
    gold_item_non_batch_pad = (gold_item != VOCAB.pad_id).any(dim=-1)
    gold_item = gold_item[gold_item_non_batch_pad]
    creator = PDB_Creator(pred_coords.detach().numpy(), seq=VOCAB.indices2aa_seq(src_seq.cpu().detach().numpy()))
    creator.save_pdb(f"data/logs/structures/{args.name}_pred.pdb", title="pred")
    creator.save_gltf(f"data/logs/structures/{args.name}_pred.gltf")
    gold_item[torch.isnan(gold_item)] = 0
    t_creator = PDB_Creator(gold_item.cpu().detach().numpy(), seq=VOCAB.indices2aa_seq(src_seq.cpu().detach().numpy()))
    t_creator.save_pdb(f"data/logs/structures/{args.name}_true.pdb", title="true")
    t_creator.save_gltfs(f"data/logs/structures/{args.name}_true.pdb", f"data/logs/structures/{args.name}_pred.pdb")
    wandb.log({"structure_comparison": wandb.Object3D(f"data/logs/structures/{args.name}_true_pred.gltf")}, commit=False)
    wandb.log({"structure_prediction": wandb.Object3D(f"data/logs/structures/{args.name}_pred.gltf")}, commit=True)


def do_eval_epoch_logging(metrics, d_loss, ln_d_loss, m_loss, c_loss, r_loss, src_seq, args, pbar, mode):
    """
    Performs all necessary logging at the end of an evaluation batch.
    Updates custom metrics dictionary and wandb logs. Prints status of training.
    """
    metrics = update_metrics(metrics, mode, d_loss, ln_d_loss, m_loss, c_loss, src_seq, r_loss, batch_level=False)
    wandb.log({f"{mode.title()} Epoch RMSE": np.sqrt(m_loss.item()),
               f"{mode.title()} Epoch RMSD": r_loss,
               f"{mode.title()} Epoch DRMSD": d_loss,
               f"{mode.title()} Epoch ln-DRMSD": ln_d_loss,
               f"{mode.title()} Epoch Combined Loss": c_loss,
               f"{mode.title()} Epoch Speed": metrics[mode]["speed"]})
    print_eval_batch_status(args, (pbar, d_loss, mode, m_loss, c_loss))


def init_metrics(args):
    """
    Returns an empty metric dictionary for recording model performance.
    """
    metrics = {"train": {"epoch-history-drmsd": [],
                         "epoch-history-combined": [],
                         "epoch-history-ln-drmsd": [],
                         "epoch-history-mse": []},
               "valid": {"epoch-history-drmsd": [],
                         "epoch-history-combined": [],
                         "epoch-history-ln-drmsd": [],
                         "epoch-history-mse": []},
               "test":  {"epoch-history-drmsd": [],
                         "epoch-history-combined": [],
                         "epoch-history-ln-drmsd": [],
                         "epoch-history-mse": []},
               "history-lr": [],
               "epoch_last_improved": -1,
               "best_valid_loss_so_far": np.inf,
               "last_chkpt_time": time.time()
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
    num_res = (src_seq != VOCAB.pad_id).sum().item()
    metrics[mode]["speed"] = num_res / (time.time() - metrics[mode]["batch-time"])
    if "speeds" not in metrics[mode].keys():
        metrics[mode]["speeds"] = []
    metrics[mode]["speeds"].append(metrics[mode]["speed"])

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
    metrics[mode]["epoch-rmsd"] /= n_batches
    metrics[mode]["epoch-history-combined"].append(metrics[mode]["epoch-combined"])
    metrics[mode]["epoch-history-drmsd"].append(metrics[mode]["epoch-drmsd"])
    metrics[mode]["epoch-history-mse"].append(metrics[mode]["epoch-mse"])
    metrics[mode]["epoch-history-ln-drmsd"].append(metrics[mode]["epoch-ln-drmsd"])
    return metrics


def prepare_log_header(args):
    """
    Returns the column ordering for the logfile.
    """
    if args.loss == "combined":
        return 'drmsd,ln_drmsd,rmse,rmsd,combined,lr,mode,granularity,time,speed\n'
    else:
        return 'drmsd,ln_drmsd,rmse,rmsd,lr,mode,granularity,time,speed\n'


class EarlyStoppingCondition(Exception):
    """
    An exception to raise when Early Stopping conditions are met.
    """
    def __init__(self, *args):
        super().__init__(*args)
