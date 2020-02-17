from tqdm import tqdm

import protein_transformer
import sys

from protein_transformer.dataset import prepare_dataloaders, MAX_SEQ_LEN
from protein_transformer.train import create_parser, load_model, setup_model_optimizer_scheduler, init_worker_pool, \
    train, init_metrics, reset_metrics_for_epoch, get_losses
import torch
import wandb

def test_batch_size(args):
    # Prepare torch
    drmsd_worker_pool = init_worker_pool(args)

    # Load dataset
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings']["max_len"]

    # Prepare model
    device = torch.device('cuda' if args.cuda else 'cpu')
    model, optimizer, scheduler = setup_model_optimizer_scheduler(args, device)

    training_data, training_eval_loader, validation_datasets, test_data = prepare_dataloaders(data, args, MAX_SEQ_LEN, use_largest_bin=True)

    del data

    return first_train_epoch(model, training_data, optimizer, device, args, pool=drmsd_worker_pool)


def first_train_epoch(model, training_data, optimizer, device, args, pool=None):
    """
    Complete 3 training steps to ensure model can train.
    """
    print(f"Testing batch size {args.batch_size}", end="")
    model.train()
    batch_iter = training_data
    max_step = 3
    for step, batch in enumerate(batch_iter):
        if step == max_step:
            break
        optimizer.zero_grad()
        src_seq, tgt_ang, tgt_crds = map(lambda x: x.to(device), batch)
        pred = model(src_seq, tgt_ang)
        loss, d_loss, ln_d_loss, m_loss, c_loss = get_losses(args, pred, tgt_ang, tgt_crds, src_seq, pool=pool, log=False)

        # Clip gradients
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # Update parameters
        optimizer.step()
        print(".", end="")

    print("success.")
    return True


if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument("--experimental_batch_size", type=int)
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert "_" not in args.name, "Please do not use a '_' in your model name. " \
                                 "Conflicts with structure files."
    args.buffering_mode = 1
    args.es_mode, args.es_metric = args.early_stopping_metric.split("-")
    args.add_sos_eos = args.model == "enc-dec"
    args.bins = "auto" if args.bins == -1 else args.bins
    args.batch_size = args.experimental_batch_size

    result = test_batch_size(args)
    if result:
        exit(0)
    else:
        exit(1)
