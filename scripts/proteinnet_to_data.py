"""
This script takes a PN processed dictionary, parses out the ID, downloads the corresponding file from the PDB,
then makes a new data dictionary, akin to the one made by data_aq.
"""

import torch
import argparse


def main():
    d = torch.load(args.input_pn_dict)

    for pnid, data in d.items():
        try:
            pdb_id, model_id, chain_id = pnid.split("_")
        except ValueError:
            continue
        print(pdb_id, model_id, chain_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses the ProteinNet dictionary for PDB IDs so they may be "
                                                 "downloaded and processed for the all-atom ProteinTransformer.")
    parser.add_argument('input_pn_dict', type=str, help='Path to PN-parsed dictionary file')
    args = parser.parse_args()
    main()

