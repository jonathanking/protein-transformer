class IncompleteStructureError(Exception):
    """An exception to raise when a structure is incomplete."""
    def __init__(self, message):
        self.message = message


class NonStandardAminoAcidError(Exception):
    """An exception to raise when a structure contains a Non-standard amino acid."""
    def __init__(self, *args):
        super().__init__(*args)

#TODO The one instance where this was called in stucture_utils.py was commented out. May not be needed.
class MissingBackboneAtomsError(Exception):
    """An exception to raise when a protein backbone is incomplete."""
    def __init__(self, message):
        self.message = message


class SequenceError(Exception):
    """An exception to raise when a sequence is not as expected."""
    def __init__(self, *args):
        super().__init__(*args)


class ContigMultipleMatchingError(Exception):
    """An exception to raise when a sequence is ambiguous due to multiple matching contig locations."""
    def __init__(self, *args):
        super().__init__(*args)


class ShortStructureError(Exception):
    """An exception to raise when a sequence too short to be meaningful."""
    def __init__(self, *args):
        super().__init__(*args)


class MissingAtomsError(Exception):
    """An exception to raise when a residue is missing atoms and bond angles can't be calculated."""
    def __init__(self, *args):
        super().__init__(*args)


class NoneStructureError(Exception):
    """An exception to raise when a parsed structure becomes None."""
    def __init__(self, *args):
        super().__init__(*args)