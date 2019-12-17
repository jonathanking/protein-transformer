""" This script will manage and create datasets of interest. For instance,
 there will be a dataset of just helices, a downsampled full dataset, etc. All
 that's necessary to provide is a .txt file in data/development listing the
 ProteinNet or PDB ID of interest. """
import sys
import torch
from glob import glob
from protein_transformer.dataset import VALID_SPLITS


def make_dev_dataset(data, dev_ids):
    # Initialize empty dictionary
    new_data  = {"train": {"ang": [], "ids": [], "crd": [], "seq": []},
                 "test":  {"ang": [], "ids": [], "crd": [], "seq": []},
                 "pnid": {}}
    d = {f"valid-{x}": {"ang": [], "ids": [], "crd": [], "seq": []} for x in VALID_SPLITS}
    new_data.update(d)

    # Add each id to every subset in the new dataset dictionary
    for did in dev_ids:
        for subdict in ["train", "test"] + [f"valid-{split}" for split in VALID_SPLITS]:
            target_subdict, target_idx = data["pnid"][did]["subset"], data["pnid"][did]["idx"]
            new_data[subdict]["seq"].append(data[target_subdict]["seq"][target_idx])
            new_data[subdict]["ang"].append(data[target_subdict]["ang"][target_idx])
            new_data[subdict]["crd"].append(data[target_subdict]["crd"][target_idx])
            new_data[subdict]["ids"].append(did)
        new_data["pnid"][did] = {"idx": len(new_data["train"]["seq"]) -1, "subset": "train"}

    # Copy any remaining data from the original dictionary
    other_items = {k: v for k, v in d.items() if k not in ["train", "test"] + [f"valid-{split}" for split in VALID_SPLITS]}
    new_data.update(other_items)

    return new_data







if __name__ == "__main__":
    _, dataset = sys.argv
    data = torch.load(dataset)
    dev_datasets = glob("../data/development/*.txt")
    for dev_dataset_file in dev_datasets:
        with open(dev_dataset_file, "r") as f:
            dev_dataset_ids = f.read().splitlines()
        new_dataset = make_dev_dataset(data, dev_dataset_ids)
        torch.save(new_dataset, dev_dataset_file.replace(".txt", ".pt"))


