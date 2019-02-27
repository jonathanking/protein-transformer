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
import os

if not len(sys.argv) == 3:
    print("Improper Usage!")
    print("pdb_generator.py <pickle dic of angles> <out_dir>")
    sys.exit()

outdir = sys.argv[-1]
pathPDBFolder(outdir)

with open(sys.argv[1], "rb") as f:
    dic = pickle.load(f)

print('Pickle successfully found and loaded')

with open(os.path.join(outdir, "order.txt"), "w") as order_file:
    for key in dic.keys():
        order_file.write(key + "\n")
        new_coords, loss, loss_norm = dic[key]
        pdb_id = key.split('_')[0]
        chain_id = key.split("_")[-1]

        prot = parsePDB(pdb_id)
        print(pdb_id, chain_id)

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
            writePDB(os.path.join(outdir,  pdb_id + '_pred_{0:.2f}_n{1:.2f}.pdb'.format(loss, loss_norm)), backbone)

