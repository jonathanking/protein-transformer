"""
Given a Pymol session with many predictions containing patterned names
    i.e. (pred_2, true_2, pred_3, true_3)

this will group together each true, pred pair with their number and the rmsd
between them.

Assumes that the objects are ordered as listed above.
"""
from pymol import cmd

objs = cmd.get_object_list("all")

i = 0
while i < len(objs) - 1:
    true, pred = objs[i], objs[i+1]
    num = true.split("_")[1] if "_" in true else "1"
    print(true, pred, num)

    rmsd, *_ = cmd.align(pred, true)

    cmd.group(f"{num}_{rmsd:.2f}", members=f"{true} {pred}")

    i += 2

