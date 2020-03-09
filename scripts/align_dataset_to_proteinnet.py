"""
This file will implement functions that allow the merging of sidechain data with
Mohammed AlQuraishi's ProteinNet. It works by conforming the data I have
generated with sidechain information to match the sequence and mask reported by
ProteinNet.

Author: Jonathan King
Date : 3/09/2020
"""

from Bio import Align
import numpy as np
import torch
from tqdm import tqdm

def init_aligner():
    """
    Creates an aligner whose weights penalize excessive gaps, make gaps
    in the ProteinNet sequence impossible, and prefer gaps at the tail ends
    of sequences.
    """
    a = Align.PairwiseAligner()

    # Don't allow for gaps or mismatches with the target sequence
    a.target_gap_score = -np.inf
    a.mismatch = -np.inf
    a.mismatch_score = -np.inf

    # Do not let matching items overwhelm determining where gaps should go
    a.match = 10

    # Generally, prefer to extend gaps than to create them
    a.query_extend_gap_score = 99
    a.query_open_gap_score = 49

    # Set slight preference for open gaps on the edges, however, if present, strongly prefer single edge gaps
    a.query_end_open_gap_score = 50
    a.query_end_extend_gap_score = 100

    return a

def get_mask_from_alignment(al):
    """ For a single alignment, return the mask as a string of '+' and '-'s. """
    alignment_str = str(al).split("\n")[1]
    return alignment_str.replace("|", "+")


def can_be_directly_merged(aligner, pn_seq, my_seq, pn_mask):
    """
    Returns True iff when pn_seq and my_seq are aligned, the resultant mask
    is the same as reported by ProteinNet. Also returns the computed_mask that
    matches with ProteinNet
    """
    a = aligner.align(pn_seq, my_seq)
    if len(a) == 0:
        return None, None

    elif len(a) == 1:
        a0 = a[0]
        computed_mask = get_mask_from_alignment(a0)
        return computed_mask == pn_mask, computed_mask

    elif len(a) > 1:
        best_mask = None
        found_a_match = False
        for a0 in a:
            computed_mask = get_mask_from_alignment(a0)
            if not best_mask:
                best_mask = computed_mask
            if computed_mask == pn_mask:
                found_a_match = True
                best_mask = computed_mask
                break
        return found_a_match, best_mask


def binary_mask_to_str(m):
    """
    Given an iterable or list of 1s and 0s representing a mask, this returns
    a string mask with '+'s and '-'s.
    """
    m = list(map(lambda x: "-" if x == 0 else "+", m))
    return "".join(m)


def find_how_many_entries_can_be_directly_merged():
    d = torch.load("/home/jok120/protein-transformer/data/proteinnet/casp12_200218_30.pt")
    pn = torch.load("/home/jok120/proteinnet/data/casp12/torch/training_30.pt")
    aligner = init_aligner()
    total = 0
    successful = 0
    with open("merging_problems.csv", "w") as f, open("merging_success.csv", "w") as sf:
        for i, my_id in enumerate(tqdm(d["train"]["ids"])):
            my_seq, pn_seq, pn_mask = d["train"]["seq"][i], pn[my_id]["primary"], binary_mask_to_str(pn[my_id]["mask"])
            result, computed_mask = can_be_directly_merged(aligner, pn_seq, my_seq, pn_mask)
            if result:
                successful += 1
                sf.write(",".join([my_id, my_seq, computed_mask]) + "\n")
            else:
                f.write(",".join([my_id, my_seq, pn_seq, pn_mask, str(result)]) + "\n")
            total += 1
        print(f"{successful} out of {total} ({successful/total}) sequences can be merged successfully.")


if __name__ == '__main__':
    find_how_many_entries_can_be_directly_merged()
