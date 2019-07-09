class IncompleteStructureError(Exception):
    """An exception to raise when a structure is incomplete."""
    # TODO Read best practices for creating Exceptions
    def __init__(self, message):
        self.message = message


class NonStandardAminoAcidError(Exception):
    """An exception to raise when a structure contains a Non-standard amino acid."""

    def __init__(self, *args):
        super().__init__(*args)