""" This script will extract matching PDB ids from an already processed data dictionary."""

import numpy as np
import torch
import sys

VALID_SPLITS = [70]


def extract_ids(d, ids_subset):
    """ Given a Python dictionary containing a ProteinNet dataset and a list of tuples
        specifying (PDB_ID, {train/test/val}), this function selects the matching
        items from the total dataset. This downsizing can be used to make smaller
        datasets for debugging and evaluation.
    """
    new_d = {"train": {"ang": [], "ids": [], "crd": [], "seq": []},
             "valid": {x: {"ang": [], "ids": [], "crd": [], "seq": []} for x in VALID_SPLITS},
             "test":  {"ang": [], "ids": [], "crd": [], "seq": []}}
    completed = 0
    for id, target_subset in ids_subset:
        ang, ids, crd, seq = get_proteinnet_data(d, id)
        if ang is None:
            continue
        if target_subset == "all":
            for subset in [new_d["train"], new_d["test"]] + [new_d["valid"][split] for split in VALID_SPLITS]:
                subset["ang"].append(ang)
                subset["ids"].append(ids)
                subset["crd"].append(crd)
                subset["seq"].append(seq)
            completed += 1
        else:
            raise NotImplementedError("Specifying different data subsets other than 'all' is not yet supported.")
    print(f"{completed} structures extracted.")
    other_items = {k: v for k, v in d.items() if k not in ["train", "test", "valid"]}
    new_d.update(other_items)
    return new_d

def get_proteinnet_data(d, id):
    for subset_name, subset in zip(["train", "valid", "test"], (d["train"], d["valid"][70], d["test"])):
        for idx, pn_id in enumerate(subset["ids"]):
            if id.upper() in pn_id:
                return subset["ang"][idx], subset["ids"][idx], subset["crd"][idx], subset["seq"][idx]
    print(f"Could not find {id}.")
    return None, None, None, None


def downsample_list(l, ids):
    return [l[i] for i in ids]

if __name__ == "__main__":
    try:
        _, inpath, outpath, id_file = sys.argv
    except:
        print("Please provide the input/output paths of the ProteinNet dataset you'd like to downsample and"
              " the path to the text file containing the PDB ids to extract.")
    d = torch.load(inpath)
    with open(id_file, "r") as f:
        ids_subset_txt = f.read().splitlines()
        ids_subset = [x.split() for x in ids_subset_txt]
    dsmall = extract_ids(d, ids_subset)
    torch.save(dsmall, outpath)
    print(f"Downsampled data written to {outpath}.")

