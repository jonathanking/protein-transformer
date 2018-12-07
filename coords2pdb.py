#!/usr/bin/env python

'''
This script generates pdb files from atomic coords. It also downloads
the corresponding true PDBs (in gzipped form).

Dependencies: ProDy, Pickle, numpy, sys

Usaage: pdb_generator.py <pickled dic>
        where dic ->  PDBID: np matrix of angles.

        The angle matrix is the 3D cords of the backbone (N-Ca-C)
'''

from prody import *
import pickle
import sys

if not len(sys.argv) == 2:
    print("Improper Usage!")
    print("pdb_generator.py <pickle dic of angles>")
    sys.exit()

with open(sys.argv[1], "rb") as f:
    dic = pickle.load(f)

print('Pickle successfully found and loaded')

for key in dic.keys():

    new_coords = dic[key]
    pdb_id = key.split('_')[0]
    chain_id = key.split("_")[-1]

    prot = parsePDB(pdb_id)

    #dealing with multiple coordsets
    if len(prot.getCoordsets())>1:
        for i in range(len(prot.getCoordsets())):
            if i==0:
                pass
            else:
                prot.delCoordset(-1)

    backbone = prot.select('protein and chain ' + chain_id + ' and name N CA C')

    if not backbone.getCoords().shape == new_coords.shape:
        # there is an error in the process!!
        print('Error! Shape mismatch for ' + str(key))
    else:
        backbone.setCoords(new_coords)
        writePDB(pdb_id + '_pred.pdb', backbone)