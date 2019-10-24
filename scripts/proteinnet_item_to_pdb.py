""" This script will extract matching PDB ids from an already processed data dictionary and generate its PDB file."""

import numpy as np
import torch
import sys
import os
sys.path.append(os.path.realpath('.'))
from protein.PDB_Creator import *
from scripts.utils.structure_utils import onehot_to_seq

VALID_SPLITS = [70]


def get_proteinnet_data(d, id):
    for subset_name, subset in zip(["train", "valid", "test"], (d["train"], d["valid"][70], d["test"])):
        for idx, pn_id in enumerate(subset["ids"]):
            if id.upper() in pn_id:
                return subset["ang"][idx], subset["ids"][idx], subset["crd"][idx], subset["seq"][idx]
    print(f"Could not find {id}.")
    return None, None, None, None


if __name__ == "__main__":
    try:
        _, inpath, outpath, pdbid = sys.argv
    except:
        print("Please provide the input path of the ProteinNet dataset you'd like to extract from,"
              " the path to the PDB file to be created, and the PDB id you'd like to generate.")
        sys.exit(1)
    d = torch.load(inpath)
    ang, ids, crd, seq = get_proteinnet_data(d, pdbid)
    seq = onehot_to_seq(seq)
    creator = PDB_Creator(crd, seq)
    creator.save_pdb(outpath, title=pdbid)

