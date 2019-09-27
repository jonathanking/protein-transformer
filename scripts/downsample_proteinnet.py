import numpy as np
import torch
import sys

VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]

def down_sample_data(d, n=96):
    """ Given a Python dictionary containing a ProteinNet dataset and a target size, n,
        this function selects n items from the total dataset at random. This downsizing
        can be used to make smaller datasets for debuggin. """
    new_subsets = []
    for subset in [d["train"], d["test"]] + [d["valid"][split] for split in VALID_SPLITS]:
        num_items = len(subset["seq"])
        num_ids = n if num_items > n else num_items
        ids = np.random.choice(np.arange(0, num_items), size=num_ids, replace=False)
        new_subset_dict = {"ang": downsample_list(subset["ang"], ids),
                           "ids": downsample_list(subset["ids"], ids),
                           "crd": downsample_list(subset["crd"], ids),
                           "seq": downsample_list(subset["seq"], ids)}

        new_subsets.append(new_subset_dict)
    new_d = {"train": new_subsets[0],
             "test": new_subsets[1],
             "valid": {split: new_subsets[2 + i] for (i, split) in enumerate(VALID_SPLITS)}}
    other_items = {k: v for k, v in d.items() if k not in ["train", "test", "valid"]}
    new_d.update(other_items)
    return new_d

def downsample_list(l, ids):
    return [l[i] for i in ids]

if __name__ == "__main__":
    try:
        _, inpath, outpath, n = sys.argv
    except:
        print("Please provide the input/output paths of the ProteinNet dataset you'd like to downsample and"
              " the number of items to select.")
    d = torch.load(inpath)
    dsmall = down_sample_data(d, int(n))
    torch.save(dsmall, outpath)
    print(f"Downsampled data written to {outpath}.")

