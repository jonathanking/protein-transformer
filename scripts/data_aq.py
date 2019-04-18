import datetime
import sys

import prody as pr
import requests
import tqdm
sys.path.append("/home/jok120/sml/proj/attention-is-all-you-need-pytorch/")
from transformer.Sidechains import SC_DATA
pr.confProDy(verbosity='error')
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import multiprocessing
import sys
import argparse

parser = argparse.ArgumentParser(description="Searches through a query of PDBs and parses/downloads chains")
parser.add_argument('query_file', type=str, help= 'Path to query file')
parser.add_argument('-o', '--out_file', type=str, help='Path to output file (.pkl file)')
args = parser.parse_args()

    
AA_MAP = {'A': 15,'C': 0,'D': 1,'E': 17,'F': 8,'G': 10,'H': 11,'I': 5,'K': 4,'L': 12,'M': 19,'N': 9,'P': 6,'Q': 3,'R': 13,'S': 2,'T': 7,'V': 16,'W': 14,'Y': 18}
CUR_DIR = "/home/jok120/pdb/"
pr.pathPDBFolder(CUR_DIR )
np.set_printoptions(suppress=True) # suppresses scientific notation when printing
np.set_printoptions(threshold=np.nan) # suppresses '...' when printing


today = datetime.datetime.today()
suffix = today.strftime("%m%d%y")
print(suffix)

if not args.out_file:
    args.out_file  = "data/data_" + suffix + ".pkl"

print ("Num arguments: ", len(sys.argv))
# obtain query from file
url = 'http://www.rcsb.org/pdb/rest/search'
#fname = sys.argv[1] #args.path, path/to/file
fname = args.query_file
with open(fname, "r") as qf:
    desc = qf.readline()
    query = qf.read()
print (desc)
print (query)
newfile = open("newfile.txt", "w")
newfile.write(query)


# Helix Only Dataset


header = {'Content-Type': 'application/x-www-form-urlencoded'}
response = requests.post(url, data=query, headers=header)
if response.status_code != 200:
    print ("Failed to retrieve results.")
    
PDB_IDS = response.text.split("\n")    
print ("Retrieved {0} PDB IDs.".format(len(PDB_IDS)))

# Set amino acid encoding and angle downloading methods
def angle_list_to_sin_cos(angs, reshape=True):
    # """ Given a list of angles, returns a new list where those angles have
#         been turned into their sines and cosines. If reshape is False, a new dim.
#         is added that can hold the sine and cosine of each angle, 
#         i.e. (len x #angs) -> (len x #angs x 2). If reshape is true, this last
#         dim. is squashed so that the list of angles becomes
#         [cos sin cos sin ...]. """
    new_list = []
    new_pad_char = np.array([1, 0])
    for a in angs:   
        new_mat = np.zeros((a.shape[0], a.shape[1], 2))
        new_mat[:, :, 0] = np.cos(a)
        new_mat[:, :, 1] = np.sin(a)
#         new_mat = (new_mat != new_pad_char) * new_mat
        if reshape:
            new_list.append(new_mat.reshape(-1, 22))
        else:
            new_list.append(new_mat)
    return new_list

def seq_to_onehot(seq):
    # """ Given an AA sequence, returns a vector of one-hot vectors."""
    vector_array = []
    for aa in seq:
        one_hot = np.zeros(len(AA_MAP), dtype=bool)
        one_hot[AA_MAP[aa]] = 1
        vector_array.append(one_hot)
    return np.asarray(vector_array)
    
#get bond angles  
def get_bond_angles(res, next_res):
   #  """ Given 2 residues, returns the ncac, cacn, and cnca bond angles between them."""
    atoms = res.backbone.copy()
    atoms_next = next_res.backbone.copy()
    ncac = pr.calcAngle(atoms[0], atoms[1], atoms[2], radian=True)
    cacn = pr.calcAngle(atoms[1], atoms[2], atoms_next[0], radian=True)
    cnca = pr.calcAngle(atoms[2], atoms_next[0], atoms_next[1], radian=True)
    return ncac, cacn, cnca
    
#get angles from chain     
def get_angles_from_chain(chain, pdb_id):
     # Given a ProDy Chain object (from a Hierarchical View), return a numpy array of 
#         angles. Returns None if the PDB should be ignored due to weird artifacts. Also measures
#         the bond angles along the peptide backbone, since they account for significat variation.
#         i.e. [[phi, psi, omega, ncac, cacn, cnca, chi1, chi2, chi3, chi4, chi5], [...] ...] """
    PAD_CHAR = 0
    OUT_OF_BOUNDS_CHAR = 0
    dihedrals = []
    sequence = ""
    
    try:
        if chain.nonstdaa:
            print("Non-standard AAs found.")
            return None
        sequence = chain.getSequence()
        length = len(sequence)
        chain = chain.select("protein and not hetero").copy()
    except Exception as e:
        print("Problem loading sequence.", e)
        return None

    all_residues = list(chain.iterResidues())
    prev = all_residues[0].getResnum()
    for i, res in enumerate(all_residues):   
        if (not res.isstdaa):
            print("Found a non-std AA. Why didn't you catch this?", chain)
            print(res.getNames())
            return None
        if res.getResnum() != prev:
            print('\rNon-continuous!!', pdb_id, end="")
            return None
        else:
            prev = res.getResnum() + 1
        try:
            phi = pr.calcPhi(res, radian=True, dist=None)
        except:
            phi = OUT_OF_BOUNDS_CHAR
        try:
            psi = pr.calcPsi(res, radian=True, dist=None)
        except:
            psi = OUT_OF_BOUNDS_CHAR
        try:
            omega = pr.calcOmega(res, radian=True, dist=None)
        except:
            omega = OUT_OF_BOUNDS_CHAR
#         if phi == 0 and psi == 0 and omega == 0:
#             return None
            
        if i == len(all_residues) - 1:
            BONDANGLES = [0, 0, 0]
        else:
            try:
                BONDANGLES = list(get_bond_angles(res, all_residues[i+1]))
            except Exception as e:
                print("Bond angle issue with", pdb_id, e)
                return None

        BACKBONE = [phi,psi,omega]
                  
        def compute_single_dihedral(atoms):
            return pr.calcDihedral(atoms[0],atoms[1],atoms[2],atoms[3],radian=True)
        
        def compute_all_res_dihedrals(atom_names):
            atoms = [res.select("name " + an) for an in atom_names]
            if None in atoms:
                return None
            res_dihedrals = []
            if len(atom_names) > 0:
                for i in range(len(atoms)-3):      
                    a = atoms[i:i+4]
                    res_dihedrals.append(compute_single_dihedral(a))
            return BACKBONE + BONDANGLES + res_dihedrals + (5 - len(res_dihedrals))*[PAD_CHAR]
        # atom_names2 = ["CA", "C"] + SC_DATA[res.getResname()]
        if res.getResname()=="ARG":
            atom_names = ["CA","C","CB","CG","CD","NE","CZ","NH1"]
        elif res.getResname()=="HIS":
            atom_names = ["CA","C","CB","CG","ND1"]
        elif res.getResname()=="LYS":
            atom_names = ["CA","C","CB","CG","CD","CE","NZ"]
        elif res.getResname()=="ASP":
            atom_names = ["CA","C","CB","CG","OD1"]
        elif res.getResname()=="GLU":
            atom_names = ["CA","C","CB","CG","CD","OE1"]
        elif res.getResname()=="SER":
            atom_names = ["CA","C","CB", "OG"]
        elif res.getResname()=="THR":
            atom_names = ["CA","C","CB","CG2"]
        elif res.getResname()=="ASN":
            atom_names = ["CA","C","CB","CG","ND2"]
        elif res.getResname()=="GLN":
            atom_names = ["CA","C","CB","CG","CD","NE2"]
        elif res.getResname()=="CYS":
            atom_names = ["CA","C","CB","SG"]
        elif res.getResname()=="GLY":
            atom_names = []
        elif res.getResname()=="PRO":
            atom_names = []
        elif res.getResname()=="ALA":
            atom_names = []
        elif res.getResname()=="VAL":
            atom_names = ["CA","C","CB","CG1"]
        elif res.getResname()=="ILE":
            atom_names = ["CA","C","CB","CG1","CD1"]
        elif res.getResname()=="LEU":
            atom_names = ["CA","C","CB","CG","CD1"]
        elif res.getResname()=="MET":
            atom_names = ["CA","C","CB","CG","SD","CE"]
        elif res.getResname()=="PHE":
            atom_names = ["CA","C","CB","CG", "CD1"]
        elif res.getResname()=="TRP":
            atom_names = ["CA","C","CB","CG","CD1"]
        elif res.getResname()=="TYR":
            atom_names = ["CA","C","CB","CG","CD1"]

        # print(atom_names)
        # print(atom_names2)
            
        calculated_dihedrals = compute_all_res_dihedrals(atom_names)
        if calculated_dihedrals == None:
            return None
        dihedrals.append(calculated_dihedrals)

    # No normalization
    dihedrals_np = np.asarray(dihedrals)
    # Check for NaNs - they shouldn't be here, but certainly should be excluded if they are.
    if np.any(np.isnan(dihedrals_np)):
        print("NaNs found")
        return None
    return dihedrals_np, sequence
# 3a. Iterate through all chains in PDB_IDs, saving all results to disk
#Remove empty string PDB ids
PDB_IDS = list(filter(lambda x: x != "", PDB_IDS))
print(len(PDB_IDS))

# 3b. Parallelized method of downloading data
#%time
def work(pdb_id):
    pdb_dihedrals = []
    pdb_sequences = []
    ids = []
    # try:
    print ("PDB ID: " , pdb_id)
    pdb = pdb_id.split(":")
    print ("New PDB ID" , pdb)
    pdb_id = pdb[0]
    print ("real pdb: " , pdb_id)
    pdb_hv = pr.parsePDB(pdb_id).getHierView()
    #if less than 2 chains,  continue
    numChains = pdb_hv.numChains()
    if numChains > 1:
        print ("Num Chains > 1, returning None for: ", pdb_id)
        none_list = open("NoneFile.txt", "a")
        none_list.write(pdb_id + "\n")
        return None
    for chain in pdb_hv:
        chain_id = chain.getChid()
        dihedrals_sequence = get_angles_from_chain(chain, pdb_id)
        if dihedrals_sequence is None:
            continue
        dihedrals, sequence = dihedrals_sequence
        pdb_dihedrals.append(dihedrals)
        pdb_sequences.append(sequence)
        ids.append(pdb_id + "_" + chain_id)
    # except Exception as e:
    #     print("Whoops, returning where I am.", e)
    if len(pdb_dihedrals) == 0:
        return None
    else:
        return pdb_dihedrals, pdb_sequences, ids


def _foo(i):
    return work(PDB_IDS[i])


with Pool(multiprocessing.cpu_count()) as p:
    results = list(tqdm.tqdm(p.imap(_foo, range(len(PDB_IDS))), total=len(PDB_IDS)))

#4. Throw out results that are None; unpack results with multiple chains
MAX_LEN = 500
results_onehots = []
c = 0
for r in results:
    if not r:
        # PDB failed to download
        continue
    ang, seq, i = r
    if len(seq[0]) > MAX_LEN:
        continue
    for j in range(len(ang)):
        results_onehots.append((ang[j], seq_to_onehot(seq[j]), i[j]))
        c += 1
print(c, "chains successfully parsed and downloaded.")
#function for additional checks of matrices
def additional_checks(matrix):
    zeros = not np.any(matrix)
    if not np.any(np.isnan(matrix)) and not np.any(np.isinf(matrix)) and not zeros:
        return True
# 5a. Remove all one-hot (oh) vectors, angles, and sequence ids from tuples
all_ohs = []
all_angs = []
all_ids = []
for r in results_onehots:
    a, oh, i = r
    if additional_checks(oh) and additional_checks(a):
        all_ohs.append(oh)
        all_angs.append(a)
        all_ids.append(i)
ohs_ids = list(zip(all_ohs, all_ids))
# need to add various checks to the lists of matrices
# 5b. Split into train, test and validation sets. Report sizes.
X_train, X_test, y_train, y_test = train_test_split(ohs_ids, all_angs, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
print("{X,y} Train, {X,y} test, {X,y} validation set sizes:\n" + \
      str(list(map(len, [X_train, y_train, X_test, y_test, X_val, y_val]))))

# 5c. Separate PDB ID/Sequence tuples. 
X_train_labels = [x[1] for x in X_train]
X_test_labels = [x[1] for x in X_test]
X_val_labels = [x[1] for x in X_val]
X_train = [x[0] for x in X_train]
X_test = [x[0] for x in X_test]
X_val = [x[0] for x in X_val]

# 6. Create a dictionary data structure, using the sin/cos transformed angles
date = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
data = {"train": {"seq": X_train,
                  "ang": angle_list_to_sin_cos(y_train),
                  "ids": X_train_labels},
        "valid": {"seq": X_val,
                  "ang": angle_list_to_sin_cos(y_val),
                  "ids": X_val_labels},
        "test":  {"seq": X_test,
                  "ang": angle_list_to_sin_cos(y_test),
                  "ids": X_test_labels},
        "settings": {"max_len": max(map(len, all_ohs))},
        "description": {desc}, 
        "date":  {date}}
#dump data
with open(args.out_file, "wb") as f:
     pickle.dump(data, f)
     
