""" This script will manage and create datasets of interest. For instance,
 there will be a dataset of just helices, a downsampled full dataset, etc. All
 that's necessary to provide is a .txt file in data/development listing the
 ProteinNet or PDB ID of interest.

 Author: Jonathan King
 Date: 12/17/2019
"""
import datetime
import sys
import torch
from glob import glob
from protein_transformer.dataset import VALID_SPLITS
import os


def make_dev_dataset(data, dev_ids):
    # Initialize empty dictionary
    new_data  = {"train": {"ang": [], "ids": [], "crd": [], "seq": []},
                 "test":  {"ang": [], "ids": [], "crd": [], "seq": []},
                 "pnids": {}}
    d = {f"valid-{x}": {"ang": [], "ids": [], "crd": [], "seq": []} for x in VALID_SPLITS}
    new_data.update(d)

    # Add each id to every subset in the new dataset dictionary
    completed = 0
    for did in dev_ids:
        try:
            target_subdict, target_idx = data["pnids"][did]["subset"], data["pnids"][did]["idx"]
        except KeyError:
            print(f"\t{did} not found in processed data.")
            continue
        for subdict in ["train", "test"] + [f"valid-{split}" for split in VALID_SPLITS]:
            new_data[subdict]["seq"].append(data[target_subdict]["seq"][target_idx])
            new_data[subdict]["ang"].append(data[target_subdict]["ang"][target_idx])
            new_data[subdict]["crd"].append(data[target_subdict]["crd"][target_idx])
            new_data[subdict]["ids"].append(did)
        new_data["pnids"][did] = {"idx": len(new_data["train"]["seq"]) -1, "subset": "train"}
        completed += 1

    # Copy any remaining data from the original dictionary
    other_items = {k: v for k, v in data.items() if k not in ["train", "test"] + [f"valid-{split}" for split in VALID_SPLITS]}
    new_data.update(other_items)
    new_data["date"] = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    print(f"\t{completed} completed.")
    return new_data


if __name__ == "__main__":
    _, dataset = sys.argv
    data = torch.load(dataset)
    dev_datasets = glob("../data/development/*.txt")
    for dev_dataset_file in dev_datasets:
        with open(dev_dataset_file, "r") as f:
            dev_dataset_ids = f.read().splitlines()
        print(f"Processing {len(dev_dataset_ids)} ProteinNet IDs from {os.path.basename(dev_dataset_file)}.")
        new_dataset = make_dev_dataset(data, dev_dataset_ids)
        torch.save(new_dataset, dev_dataset_file.replace(".txt", ".pt"))


