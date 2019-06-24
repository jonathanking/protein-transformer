"""
This script takes a PN processed dictionary, parses out the ID, downloads the corresponding file from the PDB,
then makes a new data dictionary, akin to the one made by data_aq.
"""

import torch
import argparse
import re
import prody as pr

ASTRAL_FILE = "/home/jok120/proteinnet/data/dir.cla.scope.2.07-stable.txt"


def get_pdbid_from_astral_db(domain):
    """
    Given an ASTRAL parseable file and an ASTRAL domain name, this function returns the (pdbid, description)
    associated with it.
    """
    pattern = domain + r"\s+(?P<pdbid>\S{4})\s+(?P<desc>\S+)"
    m = re.search(pattern, ASTRAL_FILE_DATA)
    print(domain, m.group('pdbid'), m.group('desc'))
    return m.group('pdbid'), m.group('desc')


def main():
    d = torch.load(args.input_pn_dict)

    for pnid, data in d.items():
        try:
            pdb_id, model_id, chain_id = pnid.split("_")
        except ValueError:
            print(pnid)
            continue
        print(pdb_id, model_id, chain_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses the ProteinNet dictionary for PDB IDs so they may be "
                                                 "downloaded and processed for the all-atom ProteinTransformer.")
    parser.add_argument('input_pn_dict', type=str, help='Path to PN-parsed dictionary file')
    parser.add_argument("--pdb_dir", default="/home/jok120/pdb/", type=str, help="Path for ProDy-downloaded PDB files.")
    args = parser.parse_args()
    with open(ASTRAL_FILE, "r") as f:
        ASTRAL_FILE_DATA = f.read()
    pr.pathPDBFolder(args.pdb_dir)
    main()

