class IncompleteStructureError(Exception):
    """An exception to raise when a structure is incomplete."""
    # TODO Read best practices for creating Exceptions
    def __init__(self, message):
        self.message = message


class NonStandardAminoAcidError(Exception):
    """An exception to raise when a structure contains a Non-standard amino acid."""

    def __init__(self, *args):
        super().__init__(*args)


class MissingBackboneAtomsError(Exception):
    """An exception to raise when a protein backbone is incomplete."""
    def __init__(self, message):
        self.message = message


class SequenceError(Exception):
    """An exception to raise when a sequence is not as expected."""
    def __init__(self, *args):
        super().__init__(*args)