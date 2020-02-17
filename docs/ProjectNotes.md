# Project Notes

This document describes some of the finer points of this project.

## How does the program handle missing residues?
Missing atoms (and thereby missing angles and missing residues) are determined at time of dataset creation by [proteinnet2pytorch.py](../scripts/proteinnet2pytorch.py). Any missing value is replaced with `np.nan`.
When comparing a predicted structure versus a true structure, where the true structure has missing atoms or angles, the nans are first removed from the true tensor, and the corresponding predicted values from the predicted tensor are also removed. The resulting tensors are then compared.
This holds true for computing MSE loss or DRMSD loss.
Batch padding is done with 0s. So, when computing loss for a single example, the batch padding is removed first, followed by the missing values/np.nans.

## Does the program predict **every** single atom of the protein structure?
No, though all that are considered relevant are predicted.

__The following atoms are not predicted:__
* Hydrogens
* Hydroxyl oxygens of terminal residues

__The following atoms are not directly predicted, but are inferred from other predicted angles using the assumption that they are a part of a planar bond configuration:__
* Carbonyl carbons attached to the backbone
* NH2 atom of Arginine
* OD2 atom of Aspartic Acid
* OE2 atom of Glutamic Acid 
* ND2 atom of Asparagine
* NE2 atom of Glutamine

__The following atoms are part of planar rings, and their placement is hard-coded relative to previously placed atoms:__
* CE1, CZ, CE2, CD2 of Phenylalanine
* CE1, NE2, CD2 of Histidine
* NE1, CE2, CZ2, CH2, CZ3, CE3, CD2 of Tryptophan
* CE1, CZ, OH, CE2, CD2  of Tyrosine
