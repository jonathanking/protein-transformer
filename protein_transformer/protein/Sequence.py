class ProteinVocabulary(object):
    """
    Represents the 'vocabulary' of amino acids for encoding a protein sequence.
    Includes pad, sos, eos, and unknown characters as well as the 20 standard
    amino acids.
    """
    def __init__(self, add_sos_eos=False):
        self.pad_char = "_"  # Pad character
        self.unk_char = "?"  # unknown character
        self.sos_char = "<"  # SOS character
        self.eos_char = ">"  # EOS character

        self._char2int = dict()
        self._int2char = dict()

        # Extract the ordered list of 1-letter amino acid codes from the project-level AA_MAP.
        self.stdaas = map(lambda x: x[0], sorted(list(AA_MAP.items()), key=lambda x: x[1]))
        self.stdaas = "".join(filter(lambda x: len(x) == 1, self.stdaas))
        for aa in self.stdaas:
            self.add(aa)

        self.add(self.pad_char)
        self.add(self.unk_char)
        if add_sos_eos:
            self.add(self.sos_char)
            self.add(self.eos_char)

    def __getitem__(self, aa):
        return self._char2int.get(aa, self._char2int[self.unk_char])

    def __contains__(self, aa):
        return aa in self._char2int

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self._char2int)

    def __repr__(self):
        return f"ProteinVocabulary[size={len(self)}]"

    def int2char(self, id):
        return self._int2char[id]

    def int2chars(self, id):
        return ONE_TO_THREE_LETTER_MAP[self._int2char[id]]

    def add(self, aa):
        if aa not in self:
            aaid = self._char2int[aa] = len(self)
            self._int2char[aaid] = aa
            return aaid
        else:
            return self[aa]

    def str2ints(self, seq, add_sos_eos=True):
        if add_sos_eos:
            return [self["<"]] + [self[aa] for aa in seq] + [self[">"]]
        else:
            return [self[aa] for aa in seq]

    def ints2str(self, ints, include_sos_eos=False):
        seq = ""
        for i in ints:
            c = self.int2char(i)
            if include_sos_eos or (c not in [self.sos_char, self.eos_char, self.pad_char]):
                seq += c
        return seq


ONE_TO_THREE_LETTER_MAP = {"R": "ARG", "H": "HIS", "K": "LYS", "D": "ASP", "E": "GLU", "S": "SER", "T": "THR",
                           "N": "ASN", "Q": "GLN", "C": "CYS", "G": "GLY", "P": "PRO", "A": "ALA", "V": "VAL",
                           "I": "ILE", "L": "LEU", "M": "MET", "F": "PHE", "Y": "TYR", "W": "TRP"}
THREE_TO_ONE_LETTER_MAP = {v: k for k, v in ONE_TO_THREE_LETTER_MAP.items()}

AA_MAP = {'A': 0, 'C': 1, 'D': 2, 'E': 3,
          'F': 4, 'G': 5, 'H': 6, 'I': 7,
          'K': 8, 'L': 9, 'M': 10, 'N': 11,
          'P': 12, 'Q': 13, 'R': 14, 'S': 15,
          'T': 16, 'V': 17, 'W': 18, 'Y': 19}
AA_MAP_INV = {v: k for k, v in AA_MAP.items()}

for one_letter_code in list(AA_MAP.keys()):
    AA_MAP[ONE_TO_THREE_LETTER_MAP[one_letter_code]] = AA_MAP[one_letter_code]

VOCAB = ProteinVocabulary()