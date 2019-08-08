"""
    Using ProteinNet as a guide, this script creates a new dataset that adds sidechain atom information.
    It retains data splits, sequences, and masks but recomputes each structure's coordinates so that
    sidechain atoms may be recorded. It also saves the entirety of the data as a single Python dictionary.

    Author: Jonathan King
    Date:   July 21, 2019
"""

import argparse
import datetime
import multiprocessing
import os
import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np
import prody as pr
import torch
import tqdm

sys.path.append("/home/jok120/protein-transformer/scripts/utils/")
from structure_utils import angle_list_to_sin_cos, seq_to_onehot, get_seq_and_masked_coords_and_angles, \
    additional_checks, zero_runs, get_residue_mask_from_structure, parse_astral_summary_file, \
get_chain_from_astral_id, get_header_seq_from_astral_id
from proteinnet_parsing import parse_raw_proteinnet
from structure_exceptions import IncompleteStructureError, NonStandardAminoAcidError, SequenceError, ContigMultipleMatchingError, ShortStructureError

sys.path.extend("../protein/")

pr.confProDy(verbosity='error')
m = multiprocessing.Manager()
SEQUENCE_ERRORS = m.list()
MULTIPLE_CONTIG_ERRORS = m.list()
FAILED_ASTRAL_IDS = m.list()
PARSING_ERRORS = m.list()
NSAA_ERRORS = m.list()
MISSING_ASTRAL_IDS = m.list()
SHORT_ERRORS = m.list()


def get_chain_from_trainid(proteinnet_id):
    """
    Given a ProteinNet ID of a training or validation set item, this function returns the associated
    ProDy-parsed chain object.
    """
    # TODO Parse CIFs, though ProDy doesn't like to do this. Issues arise with model #s, multiple atom coords
    try:
        pdbid, model_id, chid = proteinnet_id.split("_")
        if "#" in pdbid:
            pdbid = pdbid.split("#")[1]
    except ValueError:
        # TODO Implement support for ASTRAL data; ids only contain "PDBID_domain" and are currently ignored
        try:
            pdbid, astral_id = proteinnet_id.split("_")
        except ValueError:
            FAILED_ASTRAL_IDS.append(1)
            return None
        try:
            return get_chain_from_astral_id(astral_id, ASTRAL_ID_MAPPING)
        except KeyError:
            MISSING_ASTRAL_IDS.append(1)
            return None
        except:
            FAILED_ASTRAL_IDS.append(1)
            return None
    try:
        pdb_hv = pr.parsePDB(pdbid, chain=chid).getHierView()
    except (AttributeError, pr.proteins.pdbfile.PDBParseError, OSError) as e:
        PARSING_ERRORS.append(1)
        return None
    chain = pdb_hv[chid]
    try:
        if chain.numCoordsets() > 1:
            chain.setACSIndex(int(model_id))
    except IndexError:
        pass

    try:
        assert chain.getChid() == chid, "The chain ID was not as expected."
    except AttributeError:
        pass  # This is probably a segment object
    return chain


def get_chain_from_testid(proteinnet_id):
    """
    Given a ProteinNet ID of a test set item, this function returns the associated
    ProDy-parsed chain object.
    """
    # TODO: assert existence of test/targets at start of script
    category, caspid = proteinnet_id.split("#")
    try:
        pdb_hv = pr.parsePDB(os.path.join(args.input_dir, "targets", caspid + ".pdb")).getHierView()
    except AttributeError:
        PARSING_ERRORS.append(1)
        return None
    try:
        assert pdb_hv.numChains() == 1
    except:
        print("Only a single chain should be parsed from the CASP targ PDB.")
    chain = next(iter(pdb_hv))
    return chain


def get_chain_from_proteinnetid(pdbid_chain):
    """
    Determines whether or not a PN id is a test or training id and calls the corresponding method.
    """
    # If the ProteinNet ID is from the test set
    if "TBM#" in pdbid_chain or "FM#" in pdbid_chain or "TBM-hard" in pdbid_chain or "FM-hard" in pdbid_chain:
        chain = get_chain_from_testid(pdbid_chain)
    # If the ProteinNet ID is from the train or validation set
    else:
        chain = get_chain_from_trainid(pdbid_chain)
    return chain


def get_proteinnet_seq_from_id(pnid):
    """
    Given a ProteinNet ID, this method returns the associated primary AA sequence.
    """
    if "#" not in pnid:
        true_seq = PN_TRAIN_DICT[pnid]["primary"]
    elif "TBM#" in pnid or "FM#" in pnid or "TBM-hard" in pnid:
        true_seq = PN_TEST_DICT[pnid]["primary"]
    else:
        true_seq = PN_VALID_DICT[pnid]["primary"]
    return true_seq


def get_sequence_from_pdb_header(pnid):
    try:
        pdbid, model_id, chid = pnid.split("_")
        if "#" in pdbid:
            pdbid = pdbid.split("#")[1]
        p, header = pr.parsePDB(pdbid, chain=chid, header=True)
        polymer = header[chid]
        return polymer.sequence
    except ValueError:
        # This means the pnid actually refers to an ASTRAL id
        pdbid, astral_id = pnid.split("_")
        return get_header_seq_from_astral_id(astral_id, ASTRAL_ID_MAPPING)


def work(pdbid_chain):
    """
    For a single PDB ID with chain, i.e. ('1A9U_A'), fetches that PDB chain from the PDB and
    computes its angles.
    """
    true_seq = get_proteinnet_seq_from_id(pdbid_chain)
    chain = get_chain_from_proteinnetid(pdbid_chain)
    if chain is None:
        return None
    try:
        dihedrals_coords_sequence = get_seq_and_masked_coords_and_angles(chain, true_seq)
    except NonStandardAminoAcidError:
        NSAA_ERRORS.append(1)
        return None
    except ContigMultipleMatchingError:
        MULTIPLE_CONTIG_ERRORS.append(1)
        return None
    except ShortStructureError:
        SHORT_ERRORS.append(1)
        return None
    except SequenceError:
        try:
            seq = get_sequence_from_pdb_header(pdbid_chain)
            dihedrals_coords_sequence = get_seq_and_masked_coords_and_angles(chain, seq)
        except SequenceError:
            print("Not fixed.", pdbid_chain)
            SEQUENCE_ERRORS.append(1)
            return None
        except ContigMultipleMatchingError:
            MULTIPLE_CONTIG_ERRORS.append(1)
            return None

    dihedrals, coords, sequence = dihedrals_coords_sequence

    return dihedrals, coords, sequence, pdbid_chain


def unpack_processed_results(results):
    """
    Given an iterable of processed results containing angles, sequences, and PDB IDs,
    this function separates out the components (sequences as one-hot vectors, angle matrices,
    and PDB IDs) iff all were successfully preprocessed.
    """
    all_ohs = []
    all_angs = []
    all_crds = []
    all_ids = []
    c = 0
    for r in results:
        if not r:
            # PDB failed to download
            continue
        ang, coords, seq, i = r
        oh = seq_to_onehot(seq)
        if additional_checks(oh) and additional_checks(ang) and additional_checks(coords):
            all_ohs.append(oh)
            all_angs.append(ang)
            all_crds.append(coords)
            all_ids.append(i)
            c += 1
        else:
            ERROR_FILE.write(f"{i}, numerical issue\n")
    print(f"{(c * 100) / len(results):.1f}% of chains parsed. ({c}/{len(results)})")
    return all_ohs, all_angs, all_crds, all_ids


def validate_data(data):
    """
    Performs several checks on dictionary before saving.
    """
    train_len = len(data["train"]["seq"])
    test_len = len(data["test"]["seq"])
    assert all([l == train_len
                for l in map(len, [data["train"][k]
                                   for k in ["ang", "ids", "mask", "evolutionary",
                                             "secondary"]])]), "Train lengths don't match."
    assert all([l == test_len
                for l in map(len, [data["test"][k]
                                   for k in ["ang", "ids", "mask", "evolutionary",
                                             "secondary"]])]), "Test lengths don't match."


def create_data_dict(train_seq, test_seq, train_ang, test_ang, train_crd, test_crd, train_ids, test_ids, all_validation_data):
    """
    Given split data along with the query information that generated it, this function saves the
    data as a Python dictionary, which is then saved to disk using torch.save.
    """
    train_proteinnet_dict = torch.load(os.path.join(args.input_dir, "torch", TRAIN_FILE))
    valid_proteinnet_dict = torch.load(os.path.join(args.input_dir, "torch", "validation.pt"))
    test_proteinnet_dict = torch.load(os.path.join(args.input_dir, "torch", "testing.pt"))

    # Create a dictionary data structure, using the sin/cos transformed angles
    data = {"train": {"seq": train_seq,
                      "ang": angle_list_to_sin_cos(train_ang),
                      "ids": train_ids,
                      "crd": train_crd,
                      "mask": [np.asarray(train_proteinnet_dict[_id]["mask"]) for _id in train_ids],
                      "evolutionary": [np.asarray(train_proteinnet_dict[_id]["evolutionary"]) for _id in train_ids],
                      "secondary": [train_proteinnet_dict[_id]["secondary"]
                                    if "secondary" in train_proteinnet_dict[_id].keys()
                                    else None
                                    for _id in train_ids]},
            "valid": {split: dict() for split in VALID_SPLITS},
            "test": {"seq": test_seq,
                     "ang": angle_list_to_sin_cos(test_ang),
                     "ids": test_ids,
                     "crd": test_crd,
                     "mask": [np.asarray(test_proteinnet_dict[_id]["mask"]) for _id in test_ids],
                     "evolutionary": [np.asarray(test_proteinnet_dict[_id]["evolutionary"]) for _id in test_ids],
                     "secondary": [test_proteinnet_dict[_id]["secondary"]
                                   if "secondary" in test_proteinnet_dict[_id].keys()
                                   else None
                                   for _id in test_ids]},
            "settings": {"max_len": max(map(len, train_seq + test_seq))},
            "description": {f"ProteinNet {CASP_VERSION.upper()}"},
            # To parse date later, use datetime.datetime.strptime(date, "%I:%M%p on %B %d, %Y")
            "date": {datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}}
    for split, (seq_val, ang_val, crd_val, ids_val) in all_validation_data.items():
        data["valid"][split]["seq"] = seq_val
        data["valid"][split]["ang"] = angle_list_to_sin_cos(ang_val)
        data["valid"][split]["crd"] = crd_val
        data["valid"][split]["ids"] = ids_val
        data["valid"][split]["mask"] = [np.asarray(valid_proteinnet_dict[_id]["mask"]) for _id in ids_val]
        data["valid"][split]["evolutionary"] = [np.asarray(valid_proteinnet_dict[_id]["evolutionary"]) for _id in ids_val]
        data["valid"][split]["secondary"] = [valid_proteinnet_dict[_id]["secondary"]
                                             if "secondary" in valid_proteinnet_dict[_id].keys()
                                             else None
                                             for _id in ids_val]
        data["settings"]["max_len"] = max(data["settings"]["max_len"], max(map(len, seq_val)))
        valid_len = len(data["valid"][split]["seq"])
        assert all([l == valid_len for l in map(len, [data["valid"][split][k]
                                                      for k in ["ang", "ids", "mask", "evolutionary","secondary"]])]),\
            "Valid lengths don't match."
    validate_data(data)
    return data


def group_validation_set(vset_ids):
    """
    Given a list of validation set ids, (i.e. 70#1A9U_1_A), this returns a dictionary that maps each split
    to the list of PDBS in that split.
    >>> vids = ["70#1A9U_1_A", "30#1Z3F_1_B"]
    >>> group_validation_set(vids)
    {70: "70#1A9U_1_A", 30:"30#1Z3F_1_B"}
    """
    # Because there are several validation sets, we group IDs by their seq identity for use later
    valid_ids_grouped = {k: [] for k in VALID_SPLITS}
    for vid in vset_ids:
        group = int(vid[:2])
        valid_ids_grouped[group].append(vid)
    return valid_ids_grouped


def save_data_dict(data):
    """
    Saves a Python dictionary containing all training data to disk via Pickle or PyTorch.
    """
    if not args.out_file and args.pickle:
        args.out_file = "../data/proteinnet/" + CASP_VERSION + "_" + suffix + ".pkl"
    elif not args.out_file and not args.pickle:
        args.out_file = "../data/proteinnet/" + CASP_VERSION + "_" + suffix + ".pt"
    if args.pickle:
        with open(args.out_file, "wb") as f:
            pickle.dump(data, f)
    else:
        torch.save(data, args.out_file)
    print(f"Data saved to {args.out_file}.")


def post_process_data(data):
    """
    Trims parts of sequences that are masked at the start and end only.
    """
    # For masked sequences, first remove missing residues at the start and end of the sequence.
    # Then, assert that the sequence matches the one aquired from the PDB
    for dset in [data["train"], data["test"]] + [data["valid"][split] for split in VALID_SPLITS]:
        pdb_seqs = dset["seq"]
        pn_seqs = dset["primary"]
        masks = dset["masks"]
        bad_ids = []
        for i, (pdb_s, pn_s, m) in enumerate(zip(pdb_seqs, pn_seqs, masks)):
            z = zero_runs(np.asarray(m))
            if z[-1, -1] == len(pn_s):
                pn_s = pn_s[:z[-1, 0]]  # trim end
                m = m[:z[-1, 0]]
            if z[0, 0] == 0:
                pn_s = pn_s[z[0, 1]:]  # trim start
                m = m[z[0, 1]:]
            if len(pdb_s) != len(pn_s):  # "After trimming, the PN Seq and PDB seq should be the same length."
                bad_ids.append(0)

    return data


def main():
    global PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT
    train_pdb_ids, valid_ids, test_casp_ids = parse_raw_proteinnet(args.input_dir)
    print("IDs fetched.")
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = torch.load(
        os.path.join(args.input_dir, "torch", TRAIN_FILE)), torch.load(
        os.path.join(args.input_dir, "torch", "validation.pt")), torch.load(
        os.path.join(args.input_dir, "torch", "testing.pt"))

    # Download and preprocess all data from PDB IDs
    lim = None
    with Pool(multiprocessing.cpu_count()) as p:
        train_results = list(tqdm.tqdm(p.imap(work, train_pdb_ids[:lim]), total=len(train_pdb_ids[:lim])))

    valid_result_meta = {}
    for split, vids in group_validation_set(valid_ids).items():
        with Pool(multiprocessing.cpu_count()) as p:
            valid_results = list(tqdm.tqdm(p.imap(work, vids), total=len(vids)))
        valid_result_meta[split] = valid_results

    with Pool(multiprocessing.cpu_count()) as p:
        test_results = list(tqdm.tqdm(p.imap(work, test_casp_ids), total=len(test_casp_ids)))
    print("Structures processed.")
    print_summary()

    # Unpack results
    print("Training set:\t", end="")
    train_ohs, train_angs, train_strs, train_ids = unpack_processed_results(train_results)
    for split, results in valid_result_meta.items():
        print(f"Valid set {split}%:\t", end="")
        valid_result_meta[split] = unpack_processed_results(results)
    print("Test set:\t\t", end="")
    test_ohs, test_angs, test_strs, test_ids = unpack_processed_results(test_results)

    ERROR_FILE.close()

    # Split into train, test and validation sets. Report sizes.
    data = create_data_dict(train_ohs, test_ohs, train_angs, test_angs, train_strs, test_strs, train_ids, test_ids,
                            valid_result_meta)
    # data = post_process_data(data)
    save_data_dict(data)


def print_summary():
    print(f"{sum(MISSING_ASTRAL_IDS)} ASTRAL IDs were missing from the file.")
    print(f"{sum(FAILED_ASTRAL_IDS)} ASTRAL IDs failed to download for another reason.")
    print(f"{len(MULTIPLE_CONTIG_ERRORS)} ProteinNet IDs failed because of multiple matching contigs.")
    print(f"{len(SEQUENCE_ERRORS)} ProteinNet IDs failed because of mismatching sequence errors.")
    print(f"{len(NSAA_ERRORS)} ProteinNet IDs failed because of non-std AAs.")
    print(f"{len(SHORT_ERRORS)} ProteinNet IDs failed because their length was <= 2.")
    print(f"{len(PARSING_ERRORS)} ProteinNet IDs failed because of parsing errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a ProteinNet directory of raw records into a dataset for all"
                    "atom protein structure prediction.")
    parser.add_argument('input_dir', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('-o', '--out_file', type=str, help='Path to output file (.tch file)')
    parser.add_argument("--pdb_dir", default="/home/jok120/pdb/", type=str,
                        help="Path for ProDy-downloaded PDB files.")
    parser.add_argument("-t", "--tertiary", action="store_true",
                        help="Include tertiary (coordinate-level) data.", default=True)
    parser.add_argument("-p", "--pickle", action="store_true",
                        help="Save data as a pickled dictionary instead of a torch-dictionary.")
    args = parser.parse_args()
    VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]
    TRAIN_FILE = "training_100.pt"
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = None, None, None
    ASTRAL_FILE = "/home/jok120/protein-transformer/data/dir.des.scope.2.07-stable.txt"
    ASTRAL_ID_MAPPING = parse_astral_summary_file(ASTRAL_FILE)
    pr.pathPDBFolder(args.pdb_dir)
    np.set_printoptions(suppress=True)  # suppresses scientific notation when printing
    np.set_printoptions(threshold=np.nan)  # suppresses '...' when printing
    today = datetime.datetime.today()
    suffix = today.strftime("%y%m%d")
    match = re.search(r"casp\d+", args.input_dir, re.IGNORECASE)
    assert match, "The input_dir is not titled with 'caspX'."
    CASP_VERSION = match.group(0)
    ERROR_FILE = open("error.log", "w")
    try:
        main()
    except Exception as e:
        print_summary()
        raise e
