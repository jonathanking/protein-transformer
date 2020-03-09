import sys
sys.path.append("scripts")
from align_dataset_to_proteinnet import *
# from ..align_dataset_to_proteinnet import *
import pytest

global_aligner = init_aligner()

@pytest.fixture
def aligner():
    return global_aligner

@pytest.mark.parametrize("target, mobile, mask",[
    ("AAAAAAAAGAPAAAAAAA", "AAAAAAAAAAAAAAA", "++++++++---+++++++"),
    ("STARTAAAAAAAAAGAPAAAAAA", "AAAAAAAAAAAAAAA", "-----+++++++++---++++++"),
    ("STARTAAAAAAAGAAAAPAAAAAAAAAEND", "AAAAAAAAAAAAAAAA", '-----+++++++------+++++++++---')
])
def test_get_mask_from_alignment(target, mobile, mask):
    a = init_aligner()
    a1 = a.align(target, mobile)[0]
    computed_mask = get_mask_from_alignment(a1)
    assert mask == computed_mask


@pytest.mark.parametrize("pn_seq, my_seq, pn_mask", [
    ("AAAAAAAAGAPAAAAAAA", "AAAAAAAAAAAAAAA", "++++++++---+++++++"),
    ("STARTAAAAAAAAAGAPAAAAAA", "AAAAAAAAAAAAAAA", "-----+++++++++---++++++"),
    ("STARTAAAAAAAGAAAAPAAAAAAAAAEND", "AAAAAAAAAAAAAAAA", '-----+++++++------+++++++++---')
])
def test_can_be_directly_merged(aligner, pn_seq, my_seq, pn_mask):
    assert can_be_directly_merged(aligner, pn_seq, my_seq, pn_mask)[0]

@pytest.mark.parametrize("pn_seq, my_seq, pn_mask", [
    ("AAAAAAAAGAPAAAAAAA", "AAAAAAAAAAAAAAAA", "++++++++---+++++++"),
    ("STARTAAAAAAAAAGAPAAAAAA", "AAAAAAAAAAAAAAA", "-----+++++++++---+++++-"),
    ("STARTAAAAAAAGAAAAPAAAAAAAAAEND", "AAAAAAAAAAAAAAAA", '-----+++++++--+---+++++++++---')
])
def test_not_can_be_directly_merged(aligner, pn_seq, my_seq, pn_mask):
    assert not can_be_directly_merged(aligner, pn_seq, my_seq, pn_mask)[0]