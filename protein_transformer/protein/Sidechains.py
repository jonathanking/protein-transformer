import numpy as np
import torch

NUM_PREDICTED_ANGLES = 13
NUM_BB_TORSION_ANGLES = 3
NUM_BB_OTHER_ANGLES = 3
NUM_SC_ANGLES = NUM_PREDICTED_ANGLES - (NUM_BB_OTHER_ANGLES + NUM_BB_TORSION_ANGLES)
NUM_PREDICTED_COORDS = 13

ONE_TO_THREE_LETTER_MAP = {"R": "ARG", "H": "HIS", "K": "LYS", "D": "ASP", "E": "GLU", "S": "SER", "T": "THR",
                           "N": "ASN", "Q": "GLN", "C": "CYS", "G": "GLY", "P": "PRO", "A": "ALA", "V": "VAL",
                           "I": "ILE", "L": "LEU", "M": "MET", "F": "PHE", "Y": "TYR", "W": "TRP"}

THREE_TO_ONE_LETTER_MAP = {v: k for k, v in ONE_TO_THREE_LETTER_MAP.items()}

AA_MAP = {'A': 0, 'C': 1, 'D': 2, 'E': 3,
          'F': 4, 'G': 5, 'H': 6, 'I': 7,
          'K': 8, 'L': 9, 'M': 10, 'N': 11,
          'P': 12, 'Q': 13, 'R': 14, 'S': 15,
          'T': 16, 'V': 17, 'W': 18, 'Y': 19}

for one_letter_code in list(AA_MAP.keys()):
    AA_MAP[ONE_TO_THREE_LETTER_MAP[one_letter_code]] = AA_MAP[one_letter_code]

AA_MAP_INV = {v: k for k, v in AA_MAP.items()}


def deg2rad(angle):
    """
    Converts an angle in degrees to radians.
    """
    return angle * np.pi / 180.

