"""
This script is to be used to determine the maximum batch size that can fit into GPU memory for a certain
set of hyperparameters. This is designed to be called by `train.py`, which uses the exit code of this
script in order to ascertain what the maximum batch size was.

A separate script is used because, even though PyTorch allows users to clear GPU cache, a non-zero sized
cache is left over afterwards. By moving this code into its own process, then all of the memory will be
cleared when the process completes, leaving a completely free GPU available for the parent script (`train.py`).

    Author: Jonathan King
    Date: 02/17/2020
"""

from protein_transformer.dataset import prepare_dataloaders, MAX_SEQ_LEN, BinnedProteinDataset, paired_collate_fn, SimilarLengthBatchSampler
from protein_transformer.train import create_parser, setup_model_optimizer_scheduler, get_losses, init_worker_pool
import torch

def test_batch_size(args):
    """ Increases the batch size by one, stopping with the system runs out of memory. """
    # Load dataset
    import torch
    data = torch.load(args.data)
    pool = init_worker_pool(args)

    def add_incrementer(n):
        return n + 1

    def mul_incrementer(n):
        return n * 2

    incrementer = mul_incrementer
    first = True

    args.max_token_seq_len = data['settings']["max_len"]
    for i in range(2):
        while True:
            try:
                # Prepare model
                import torch
                device = torch.device('cuda' if args.cuda else 'cpu')
                model, optimizer, scheduler = setup_model_optimizer_scheduler(args, device)

                train_dataset = BinnedProteinDataset(
                    seqs=data['train']['seq'] * args.repeat_train,
                    crds=data['train']['crd'] * args.repeat_train,
                    angs=data['train']['ang'] * args.repeat_train,
                    add_sos_eos=args.add_sos_eos, skip_missing_residues=args.skip_missing_res_train, bins=args.bins)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    num_workers=1,
                    collate_fn=paired_collate_fn,
                    batch_sampler=SimilarLengthBatchSampler(train_dataset,
                                                            args.batch_size,
                                                            dynamic_batch=args.batch_size * MAX_SEQ_LEN,
                                                            optimize_batch_for_cpus=False,
                                                            use_largest_bin=True))
                res = first_train_epoch(model, train_loader, optimizer, device, args, pool=pool)

                # Clean up
                del device
                del model, optimizer, scheduler
                del train_dataset, train_loader
                args.batch_size = incrementer(args.batch_size)
                torch.cuda.empty_cache()
                del torch
                del res
            except RuntimeError:
                print("(crash)")
                if not first:
                    return args.batch_size
                first = False
                incrementer = add_incrementer
                try:
                    del device
                    del model, optimizer, scheduler
                    del train_dataset, train_loader
                    del res
                except:
                    pass

                torch.cuda.empty_cache()
                del torch
                args.batch_size = int(args.batch_size // 2) + 1
                break


def first_train_epoch(model, training_data, optimizer, device, args, pool=None):
    """
    Complete 3 training steps to ensure model can train with the specified batch size.
    """
    print(f"Testing batch size {args.batch_size: >2} [", end="")
    model.train()
    batch_iter = training_data
    max_step = 2
    for step, batch in enumerate(batch_iter):
        if step == max_step:
            break
        optimizer.zero_grad()
        src_seq, tgt_ang, tgt_crds = map(lambda x: x.to(device), batch)
        print(f"({src_seq.shape[0]})", end="")
        pred = model(src_seq, tgt_ang)
        loss, d_loss, ln_d_loss, m_loss, c_loss = get_losses(args, pred, tgt_ang, tgt_crds, src_seq, pool=pool, log=False)

        # Clip gradients
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # Update parameters
        optimizer.step()

    print("] success.")
    return True


if __name__ == '__main__':
    parser = create_parser()
    parser.add_argument("--experimental_batch_size", type=int)
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert "_" not in args.name, "Please do not use a '_' in your model name. " \
                                 "Conflicts with structure files."
    args.buffering_mode = 1
    if not args.early_stopping_metric:
        args.early_stopping_metric = f"train-{args.loss}"
    args.es_mode, args.es_metric = args.early_stopping_metric.split("-")
    args.add_sos_eos = args.model == "enc-dec"
    args.bins = "auto" if args.bins == -1 else args.bins
    args.batch_size = args.experimental_batch_size

    result = test_batch_size(args)
    exit(result)
