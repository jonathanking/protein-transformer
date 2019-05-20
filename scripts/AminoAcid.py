from transformer.Sidechains import SC_DATA







class AminoAcid(object):
    """ A class to represent an amino acid sidechain. Its main utility is in constructing sidechains
        during structure prediction by aligning planar atoms to already predicted atoms."""

    def __init__(self, name):
        self.name = name
        self.predicted = SC_DATA[name]["pred_atoms"]

        self.alignment_frag = self.lookup_alignment_frag(name)
        self.align_to_frag = self.lookup_align_to_frag(name)

    def lookup_alignment_frag(self, name):
        """ Returns a coordinate set of the fragment to be aligned. """
        raise NotImplementedError()

    def lookup_align_to_frag(self, name):
        """ Returns a coordinate set of the fragment that will be aligned to. """
        # TODO Assert that the align-to frag is the same as the last 3 predicted atoms
        # assert result == self.predicted[-3:]
        raise NotImplementedError()

    def complete_structure(self):
        aligned_frag = self.align(self.alignment_frag, self.align_to_frag)



