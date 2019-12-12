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
import re
import sys
from multiprocessing import Pool

import numpy as np
import prody as pr
import torch
import tqdm

sys.path.append(".")
from protein_transformer.protein.structure_utils import angle_list_to_sin_cos, seq_to_onehot, get_seq_and_masked_coords_and_angles, \
    no_nans_infs_allzeros, parse_astral_summary_file, \
    get_chain_from_astral_id, get_header_seq_from_astral_id, GLOBAL_PAD_CHAR
import proteinnet_parsing
from protein_transformer.protein.structure_exceptions import IncompleteStructureError, NonStandardAminoAcidError, SequenceError, ContigMultipleMatchingError, ShortStructureError, MissingAtomsError



pr.confProDy(verbosity='error')
m = multiprocessing.Manager()
SEQUENCE_ERRORS = m.list()
MULTIPLE_CONTIG_ERRORS = m.list()
FAILED_ASTRAL_IDS = m.list()
PARSING_ERRORS = m.list()
NSAA_ERRORS = m.list()
MISSING_ASTRAL_IDS = m.list()
SHORT_ERRORS = m.list()
PARSING_ERROR_ATTRIBUTE = m.list()
PARSING_ERROR = m.list()
PARSING_ERROR_OSERROR = m.list()
UNKNOWN_EXCEPTIONS = m.list()
MISSING_ATOMS_ERROR = m.list()
NONE_CHAINS = m.list()
NO_PBD_FILE = m.list()

def get_chain_from_trainid(proteinnet_id):
    """
    Given a ProteinNet ID of a training or validation set item, this function returns the associated
    ProDy-parsed chain object. "1A9U_2_A"
    """
    # Try parsing the ID as a PDB ID. If it fails, assume it's an ASTRAL ID.
    try:
        pdbid, model_id, chid = proteinnet_id.split("_")
        if "#" in pdbid:
            pdbid = pdbid.split("#")[1]
    except ValueError:
        try:
            pdbid, astral_id = proteinnet_id.split("_")
            return get_chain_from_astral_id(astral_id.replace("-", "_"), ASTRAL_ID_MAPPING)
        except KeyError:
            MISSING_ASTRAL_IDS.append(proteinnet_id)
            return None
        except ValueError:
            FAILED_ASTRAL_IDS.append(proteinnet_id)
            return None
        except:
            FAILED_ASTRAL_IDS.append(proteinnet_id)
            return None

    # Continue loading the chain, given the PDB ID
    try:
        chain = pr.parsePDB(pdbid, chain=chid)
    except:
        try:
            chain = pr.parseCIF(pdbid, chain=chid) # changed pr.parsePDB to pr.parseCIF, removed heirarchal view
        except AttributeError:
            PARSING_ERROR_ATTRIBUTE.append(proteinnet_id)
            return None
        except pr.proteins.pdbfile.PDBParseError:
            PARSING_ERROR.append(proteinnet_id)
            return None
        except OSError:
            PARSING_ERROR_OSERROR.append(proteinnet_id)
            return None
        except Exception as e:
            UNKNOWN_EXCEPTIONS.append(str(e) + " " + proteinnet_id)
            return None

    if chain is None:
        print(proteinnet_id)
        NONE_CHAINS.append(proteinnet_id)
        return None
    # Attempt to select a coordset
    try:
        if chain.numCoordsets() > 1:
            chain.setACSIndex(int(model_id))
    except IndexError:
        # TODO is this appropriate to pass? make global variable that returns same kinda error as above. happens a lot
        pass

    return chain


def get_chain_from_testid(proteinnet_id):
    """
    Given a ProteinNet ID of a test set item, this function returns the associated
    ProDy-parsed chain object.
    """
    # TODO: assert existence of test/targets at start of script
    category, caspid = proteinnet_id.split("#")
    try:
        pdb_hv = pr.parsePDB(os.path.join(args.input_dir, "targets", caspid + ".pdb")).getHierView() # TODO change to pr.parseCIF?
    except AttributeError:
        PARSING_ERRORS.append(proteinnet_id)
        return None
    try:
        assert pdb_hv.numChains() == 1
    except:
        print("Only a single chain should be parsed from the CASP targ PDB.")
        pass
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


def work(pdbid_chain):
    """
    For a single PDB ID with chain, i.e. ('1A9U_1_A'), fetches that PDB chain from the PDB and
    computes its angles, coordinates, and sequence. The angles and coordinates contain
    GLOBAL_PAD_CHARs where there was missing data.
    """
    true_seq = get_proteinnet_seq_from_id(pdbid_chain) # TODO replace this function that returns chain from file w/ seq
    chain = get_chain_from_proteinnetid(pdbid_chain)  # Returns ProDy chain object
    if chain is None:
        return None
    try:
        dihedrals_coords_sequence = get_seq_and_masked_coords_and_angles(chain, true_seq)
    except NonStandardAminoAcidError:
        NSAA_ERRORS.append(pdbid_chain)
        return None
    except ContigMultipleMatchingError:
        MULTIPLE_CONTIG_ERRORS.append(pdbid_chain)
        return None
    except ShortStructureError:
        SHORT_ERRORS.append(pdbid_chain)
        return None
    except MissingAtomsError:
        MISSING_ATOMS_ERROR.append(pdbid_chain)
        return None
    except SequenceError:
        print("Not fixed.", pdbid_chain)
        SEQUENCE_ERRORS.append(pdbid_chain)
        return None

    # If we've made it this far, we can unpack the data and return it
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
        if no_nans_infs_allzeros(oh) and no_nans_infs_allzeros(ang) and no_nans_infs_allzeros(coords):
            all_ohs.append(oh)
            all_angs.append(ang)
            all_crds.append(coords)
            all_ids.append(i)
            c += 1
    print(f"{(c * 100) / len(results):.1f}% of chains parsed. ({c}/{len(results)})")
    return all_ohs, all_angs, all_crds, all_ids


def validate_data_dict(data):
    """
    Performs several checks on dictionary before saving.
    """
    # Assert size of each data subset matches
    train_len = len(data["train"]["seq"])
    test_len = len(data["test"]["seq"])
    items_recorded = ["seq", "ang", "ids", "crd"]
    for num_items, subset in zip([train_len, test_len], ["train", "test"]):
        assert all([l == num_items
                    for l in map(len, [data[subset][k]
                                       for k in items_recorded])]), f"{subset} lengths don't match."

    for split in VALID_SPLITS:
        valid_len = len(data["valid"][split]["seq"])
        assert all([l == valid_len for l in map(len, [data["valid"][split][k] for k in ["ang", "ids", "crd"]])]), \
            "Valid lengths don't match."


def create_data_dict(train_seq, test_seq, train_ang, test_ang, train_crd, test_crd, train_ids, test_ids, all_validation_data):
    """
    Given split data along with the query information that generated it, this function saves the
    data as a Python dictionary, which is then saved to disk using torch.save.
    See commit  d1935a0869720f85c00824f3aecbbfc6b947711c for a method that saves all relevant information.
    """
    # Create a dictionary data structure, using the sin/cos transformed angles
    data = {"train": {"seq": train_seq,
                      "ang": angle_list_to_sin_cos(train_ang),
                      "ids": train_ids,
                      "crd": train_crd},
            "valid": {split: dict() for split in VALID_SPLITS},
            "test": {"seq": test_seq,
                     "ang": angle_list_to_sin_cos(test_ang),
                     "ids": test_ids,
                     "crd": test_crd},
            "settings": {"max_len": max(map(len, train_seq + test_seq)),
                         "pad_char": GLOBAL_PAD_CHAR},
            "description": {f"ProteinNet {CASP_VERSION.upper()}"},
            # To parse date later, use datetime.datetime.strptime(date, "%I:%M%p on %B %d, %Y")
            "date": {datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}}
    max_val_len = 0
    for split, (seq_val, ang_val, crd_val, ids_val) in all_validation_data.items():
        data["valid"][split]["seq"] = seq_val
        data["valid"][split]["ang"] = angle_list_to_sin_cos(ang_val)
        data["valid"][split]["crd"] = crd_val
        data["valid"][split]["ids"] = ids_val
        max_split_len = max(data["settings"]["max_len"], max(map(len, seq_val)))
        max_val_len = max_split_len if max_split_len > max_val_len else max_val_len
    data["settings"]["max_len"] = max(list(map(len, train_seq + test_seq)) + [max_val_len])

    validate_data_dict(data)
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
    if not args.out_file:
        args.out_file = "data/proteinnet/" + CASP_VERSION + "_" + SUFFIX + ".pt"
    torch.save(data, args.out_file)
    print(f"Data saved to {args.out_file}.")


def main():
    global PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT
    train_pdb_ids, valid_ids, test_casp_ids = proteinnet_parsing.parse_raw_proteinnet(args.input_dir, TRAIN_FILE)
    print("IDs fetched.")
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = torch.load(
        os.path.join(args.input_dir, "torch", TRAIN_FILE)), torch.load(
        os.path.join(args.input_dir, "torch", "validation.pt")), torch.load(
        os.path.join(args.input_dir, "torch", "testing.pt"))
    print(len(train_pdb_ids), len(valid_ids), len(test_casp_ids))
    # Download and preprocess all data from PDB IDs
    lim = None
    train_results = []
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        train_results = list(tqdm.tqdm(p.imap(work, train_pdb_ids[:lim]), total=len(train_pdb_ids[:lim])))
    valid_result_meta = {}
    for split, vids in group_validation_set(valid_ids).items():
        valid_result_meta[split] = []
        for vid in tqdm.tqdm(vids):
            valid_result_meta[split].append(work(vid))

    test_results = []
    for tid in tqdm.tqdm(test_casp_ids):
        test_results.append(work(tid))

    print("Structures processed.")
    print_failure_summary()

    # Unpack results
    print("Training set:\t", end="")
    train_ohs, train_angs, train_strs, train_ids = unpack_processed_results(train_results)
    for split, results in valid_result_meta.items():
        print(f"Valid set {split}%:\t", end="")
        valid_result_meta[split] = unpack_processed_results(results)
    print("Test set:\t\t", end="")
    test_ohs, test_angs, test_strs, test_ids = unpack_processed_results(test_results)

    # Split into train, test and validation sets. Report sizes.
    data = create_data_dict(train_ohs, test_ohs, train_angs, test_angs, train_strs, test_strs, train_ids, test_ids,
                            valid_result_meta)
    save_data_dict(data)


def print_failure_summary():
    """ Summarizes failures associated with the processing of ProteinNet ID data. """
    print(f"{len(MISSING_ASTRAL_IDS)} ASTRAL IDs were missing from the file.")
    print(f"{len(FAILED_ASTRAL_IDS)} ASTRAL IDs failed to download for another reason.")
    print(f"{len(MULTIPLE_CONTIG_ERRORS)} ProteinNet IDs failed because of multiple matching contigs.")
    print(f"{len(SEQUENCE_ERRORS)} ProteinNet IDs failed because of mismatching sequence errors.")
    print(f"{len(NSAA_ERRORS)} ProteinNet IDs failed because of non-std AAs.")
    print(f"{len(SHORT_ERRORS)} ProteinNet IDs failed because their length was <= 2.")
    print(f"{len(PARSING_ERRORS)} ProteinNet IDs failed because of parsing errors.")
    print(f"{len(PARSING_ERROR_ATTRIBUTE)} ProteinNet IDs failed because of attribute errors.")
    print(f"{len(PARSING_ERROR)} ProteinNet IDs failed because of standard parsing error.")
    print(f"{len(PARSING_ERROR_OSERROR)} ProteinNet IDs failed because of OSError.")
    print(f"{len(UNKNOWN_EXCEPTIONS)} ProteinNet IDs failed because of unknown OSError.")
    print(f"{len(MISSING_ATOMS_ERROR)} ProteinNet IDs failed because residues were missing N, CA1, or C atoms.")
    print(f"{len(NONE_CHAINS)} ProteinNet IDs failed because no chain existed.")
    print(f"{len(NO_PBD_FILE)} ProteinNet IDs failed because no PDB file exists and they probably have .CIF instead.")


    with open('errors/MISSING_ASTRAL_IDS.txt', 'w') as f:
        f.write('\n'.join(MISSING_ASTRAL_IDS))
    with open('errors/FAILED_ASTRAL_IDS.txt', 'w') as f:
        f.write('\n'.join(FAILED_ASTRAL_IDS))
    with open('errors/MULTIPLE_CONTIG_ERRORS.txt', 'w') as f:
        f.write('\n'.join(MULTIPLE_CONTIG_ERRORS))
    with open('errors/SEQUENCE_ERRORS.txt', 'w') as f:
        f.write('\n'.join(SEQUENCE_ERRORS))
    with open('errors/NSAA_ERRORS.txt', 'w') as f:
        f.write('\n'.join(NSAA_ERRORS))
    with open('errors/SHORT_ERRORS.txt', 'w') as f:
        f.write('\n'.join(SHORT_ERRORS))
    #Total number of parsing errors, may not be needed
    with open('errors/PARSING_ERRORS.txt', 'w') as f:
        f.write('\n'.join(PARSING_ERRORS))
    #Added to split up the three category of parsing errors, this one handles attribute errors
    with open('errors/PARSING_ERROR_ATTRIBUTE.txt', 'w') as f:
        f.write('\n'.join(PARSING_ERROR_ATTRIBUTE))
    #Added to handle 'regular' parsing errors
    with open('errors/PARSING_ERROR.txt', 'w') as f:
        f.write('\n'.join(PARSING_ERROR))
    #Added to handle OS related parsing errors
    with open('errors/PARSING_ERROR_OSERROR.txt', 'w') as f:
        f.write('\n'.join(PARSING_ERROR_OSERROR))

    #Added for cases where pr.parseCIF returned unknown exception in function get_chain_from_trainid(proteinnet_id)
    with open('errors/UNKNOWN_EXCEPTION.txt', 'w') as f:
        f.write('\n'.join(UNKNOWN_EXCEPTIONS))
    #Added for cases where measure_phi_psi_omega or measure_bond_angles fail due to missing atom information
    with open('errors/MISSING_ATOMS_ERRORS.txt', 'w') as f:
        f.write('\n'.join(MISSING_ATOMS_ERROR))
    #Added when chain variable returned 'none' in function get_chain_from_trainid(proteinnet_id)
    with open('errors/NONE_CHAINS.txt', 'w') as f:
        f.write('\n'.join(NONE_CHAINS))
    #Added to handle when no PDB file existed for entry
    with open('errors/NO_PDB_FILE.txt', 'w') as f:
        f.write('\n'.join(NO_PBD_FILE))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a ProteinNet directory of raw records into a dataset for all"
                    "atom protein structure prediction.")
    parser.add_argument('input_dir', type=str, help='Path to ProteinNet raw records directory.')
    parser.add_argument('-o', '--out_file', type=str, help='Path to output file (.tch file)')
    parser.add_argument("--pdb_dir", default=os.path.expanduser("~/pdb/"), type=str,
                        help="Path for ProDy-downloaded PDB files.")
    parser.add_argument('--training_set', type=int, default=100, help='Which thinning of the training set to parse. '
                                                                      '{30,50,70,90,95,100}. Default 100.')
    args = parser.parse_args()

    VALID_SPLITS = [10, 20, 30, 40, 50, 70, 90]
    TRAIN_FILE = f"training_{args.training_set}.pt"
    PN_TRAIN_DICT, PN_VALID_DICT, PN_TEST_DICT = None, None, None
    ASTRAL_FILE = "data/proteinnet/astral_pdb_map.txt"#"data/fullDict.txt" # combined previous versions of dir.des.scope.2.xx-stable.txt into one big dict
    ASTRAL_ID_MAPPING = parse_astral_summary_file(ASTRAL_FILE)
    SUFFIX = str(datetime.datetime.today().strftime("%y%m%d")) + f"_{args.training_set}"
    match = re.search(r"casp\d+", args.input_dir, re.IGNORECASE)
    assert match, "The input_dir is not titled with 'caspX'."
    CASP_VERSION = match.group(0)

    pr.pathPDBFolder(args.pdb_dir)  # Set PDB download location
    np.set_printoptions(suppress=True)  # suppresses scientific notation when printing
    np.set_printoptions(threshold=sys.maxsize)  # suppresses '...' when printing

    try:
        main()
    except Exception as e:
        print_failure_summary()
        raise e
