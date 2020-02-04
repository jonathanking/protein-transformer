"""
Extracts the relevant sidechain information from AMBER forcefield files so that
they can be used in sidechain construction.

See here (https://ambermd.org/FileFormats.php#topology) for information on
reading the `frcmod.ff14SB` file.

See here
(https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/PDB_format_1992.pdf)
on page 27 for information on the mapping from atom names to atom types, \
for reading the `amino12.lib` file.

See here (https://ambermd.org/doc12/Amber19.pdf) on page 33 for information on
the ff14SB AMBER force field, which all of these parameters are based on.
ff14SB itself is a modification of the parameters found in `parm10.dat`.

Note that the ff14SB force field contains no information on sidechain
conformers, but only contains torsional potential information.
"""
import re
import pprint

BUILD_ORDER = {"ALA": ["CB"],
               "ARG": ["CB","CD", "NE", "CZ", "NH1", "NH2"],
               "ASN": ["CB","CG", "OD1","ND2"],
               "ASP": ["CB","CG","OD1","OD2"],
               "CYS": ["CB", "SG"],
               "GLU": ["CB","CG", "CD", "OE1", "OE2"],
               "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
               "GLY": [],
               "HIS": ["CB", "CG", "ND1","CE1","NE2","CD2"],
               "ILE": ["CB","CG1","CD1","CG2"],
               "LEU": ["CB","CG", "CD1", "CD2"],
               "LYS": ["CB","CG", "CD","CE","NZ"],
               "MET": ["CB","CG", "SD", "CE"],
               "PHE": ["CB","CG","CD1","CE1","CZ","CE2","CD2"],
               "PRO": ["CB","CG","CD"],
               "SER": ["CB","OG"],
               "THR": ["CB","OG1","CG2"],
               "TRP": ["CB","CG","CD1","NE1","CE2","CZ2","CH2","CZ3","CE3","CD2"],
               "TYR": ["CB","CG","CD1","CE1","CZ","OH","CE2","CD2"],
               "VAL": ["CB","CG1","CG2"]}

BUILD_ORDER_CHAINS =  {"ALA": [["CB"]],
                       "ARG": [["CB","CG","CD", "NE", "CZ", "NH1"],
                               ["CB", "CG", "CD", "NE", "CZ", "NH2"]],
                       "ASN": [["CB","CG", "OD1"],
                               ["CB","CG", "ND2"]],
                       "ASP": [["CB","CG","OD1"],
                               ["CB","CG","OD2"]],
                       "CYS": [["CB", "SG"]],
                       "GLU": [["CB","CG", "CD", "OE1"],
                               ["CB","CG", "CD", "OE2"]],
                       "GLN": [["CB", "CG", "CD", "OE1"],
                               ["CB","CG", "CD", "NE2"]],
                       "GLY": [[]],
                       "HIS": [["CB", "CG", "ND1","CE1","NE2","CD2"]],
                       "ILE": [["CB","CG1","CD1"],
                               ["CB","CG2"]],
                       "LEU": [["CB","CG", "CD1"],
                               ["CB","CG", "CD2"]],
                       "LYS": [["CB","CG", "CD","CE","NZ"]],
                       "MET": [["CB","CG", "SD", "CE"]],
                       "PHE": [["CB","CG","CD1","CE1","CZ","CE2","CD2"]],
                       "PRO": [["CB","CG","CD"]],
                       "SER": [["CB","OG"]],
                       "THR": [["CB","OG1"],
                               ["CB","CG2"]],
                       "TRP": [["CB","CG","CD1","NE1","CE2","CZ2","CH2","CZ3","CE3","CD2"]],
                       "TYR": [["CB","CG","CD1","CE1","CZ","OH"],
                               ["CB","CG","CD1","CE1","CZ","CE2","CD2"]],
                       "VAL": [["CB","CG1"],
                               ["CB","CG2"]]}


def extract_atom_name_type_map(atom_name_file):
    """ Given a force field file that contains the amino acid topologies, this
        function retrieves the mapping from atom names to atom types for each
        amino acid.
        Ex: {'ALA': {'C': 'C', 'CA': 'CX', 'CB': 'CT', ... }}"""
    aa_data = {}
    regex = r"entry.(\w{3}).unit.atomspertinfo.*\n((\W.*\n)+?)!"
    with open(atom_name_file, "r") as f:
        text = f.read()
    matches = re.finditer(regex, text, re.MULTILINE)
    for m in matches:
        atom_name_to_type = {}
        for line in m.group(2).splitlines():
            atoms = line.split()[:2]
            if atoms[0].strip('"').startswith("H"):
                continue
            atom_name_to_type[atoms[0].strip('"')] = atoms[1].strip('"').ljust(2)
        aa_data[m.group(1)] = atom_name_to_type

    # TODO Which histidine naming convention is appropriate?
    # http://ambermd.org/Questions/HIS.html
    aa_data["HIS"] = aa_data["HID"]
    return aa_data


def extract_bonds_and_angle_info(force_field):
    """ Given a force field files, extracts the values use for equilibrium
    bond lengths and angles. """
    info = {"bonds":{},
            "angles": {}}
    bond_regex = r"^(.{2}-.{2})\s+\S+\w+\s+(\S+)"
    angle_regex = r"^(.{2}-.{2}-.{2})\s+\S+\w+\s+(\S+)"
    with open(force_field, "r") as f:
        text = f.read()

    # Extract bond info
    matches = re.finditer(bond_regex, text, re.MULTILINE)
    for m in matches:
        atoms = m.group(1)
        length = m.group(2)
        info["bonds"][atoms] = float(length)

    # Extract angle info
    matches = re.finditer(angle_regex, text, re.MULTILINE)
    for m in matches:
        atoms = m.group(1)
        angle = m.group(2)
        info["angles"][atoms] = float(angle)

    return info


def create_full_amino_acid_build_dict(atom_name_dict, bond_angle_dict):
    """ This function takes the information that has been gathered from the
    force field parameter files, and writes out a new dictionary that is
    an attempt to record all angle, bond, and torsion information for each
    amino acid. There are some incorrect assumptions made at this point. For
    instance, the terminal bond/angle/torsion parameters may be wrong if the
    amino acid has a branch. Therefore, this method is used as a first pass
    to create a python dictionary that will hold all of the necessary
    information for all of the amino acids. It must then be modified slightly
    by hand to account for branched amino acids (ARG, ETC)"""
    AMINO_ACID_INFO = {}
    for AA, build_order_chains in BUILD_ORDER_CHAINS.items():
        AMINO_ACID_INFO[AA] = {}
        # Add bonds
        bonds_names = []
        bonds_types = []
        bond_lens = []

        for i, chain in enumerate(build_order_chains):
            # This corresponds to a normal chain, beginning with a CA
            if i == 0:
                prev_bond_atom = "CA"
            else:
                j = 0
                while build_order_chains[0][j] == build_order_chains[1][j]:
                    j += 1
                prev_bond_atom = chain[j-1]
                chain = chain[j:]
            for atom_name in chain:
                cur_bond = [prev_bond_atom, atom_name]
                cur_bond_names = "-".join(cur_bond)
                bonds_names.append(cur_bond_names)
                cur_bond_types = "-".join([atom_name_dict[AA][an] for an in cur_bond])
                bonds_types.append(cur_bond_types)

                try:
                    bond_lens.append(bond_angle_dict["bonds"][cur_bond_types])
                except KeyError:
                    try:
                        cur_bond_types = "-".join(cur_bond_types.split("-")[::-1])
                        bond_lens.append(bond_angle_dict["bonds"][cur_bond_types])
                    except KeyError:
                        bond_lens.append("?")
                prev_bond_atom = atom_name
        AMINO_ACID_INFO[AA]["bonds-names"] = bonds_names
        AMINO_ACID_INFO[AA]["bonds-types"] = bonds_types
        AMINO_ACID_INFO[AA]["bond-lens"] = bond_lens

        angles_names = []
        angles_types = []
        angles_vals = []
        for i, chain in enumerate(build_order_chains):
            prev_2_atoms = ["N", "CA"]
            if i == 1:
                j = 0
                cur_angles = [*prev_2_atoms, chain[j]]
                while "-".join(cur_angles) in angles_names and j < len(chain)-1:
                    prev_2_atoms = [prev_2_atoms[1], chain[j]]
                    cur_angles = [*prev_2_atoms, chain[j+1]]
                    j += 1
                chain = chain[j:]


            for atom_name in chain:
                cur_angles = [*prev_2_atoms, atom_name]
                cur_angles_names = "-".join(cur_angles)
                angles_names.append(cur_angles_names)
                cur_angles_types = "-".join([atom_name_dict[AA][an] for an in cur_angles])
                angles_types.append(cur_angles_types)
                try:
                    angles_vals.append(bond_angle_dict["angles"][cur_angles_types])
                except KeyError:
                    try:
                        cur_angles_types = "-".join(cur_angles_types.split("-")[::-1])
                        angles_vals.append(bond_angle_dict["angles"][cur_angles_types])
                    except KeyError:
                        angles_vals.append("?")
                prev_2_atoms = [prev_2_atoms[-1], atom_name]
        AMINO_ACID_INFO[AA]["angles-names"] = angles_names
        AMINO_ACID_INFO[AA]["angles-types"] = angles_types
        AMINO_ACID_INFO[AA]["angles-vals"] = angles_vals

        torsion_names = []
        torsion_types = []
        torsion_vals = []
        for i, chain in enumerate(build_order_chains):
            prev_3_atoms = ["C", "N", "CA"]
            if i == 1:
                j = 0
                cur_torsion = [*prev_3_atoms, chain[j]]
                while "-".join(cur_torsion) in torsion_names and j < len(chain) - 1:
                    prev_3_atoms = [prev_3_atoms[-2], prev_3_atoms[-1], chain[j]]
                    cur_torsion = [*prev_3_atoms, chain[j + 1]]
                    j += 1
                chain = chain[j:]
            for atom_name in chain:
                cur_torsion = [*prev_3_atoms, atom_name]
                cur_torsion_names = "-".join(cur_torsion)
                torsion_names.append(cur_torsion_names)
                cur_torsion_types = "-".join(
                    [atom_name_dict[AA][an] for an in cur_torsion])
                torsion_types.append(cur_torsion_types)
                torsion_vals.append("?")
                prev_3_atoms = [prev_3_atoms[-2], prev_3_atoms[-1], atom_name]
        AMINO_ACID_INFO[AA]["torsion-names"] = torsion_names
        AMINO_ACID_INFO[AA]["torsion-types"] = torsion_types
        AMINO_ACID_INFO[AA]["torsion-vals"] = torsion_vals

    return AMINO_ACID_INFO


def main():
    with open("atom_name_dict.txt", "w") as f:
        atom_name_file = "amino12.lib"
        atom_name_dict = extract_atom_name_type_map(atom_name_file)
        f.write(pprint.pformat(atom_name_dict))

    with open("ff14sb_bonds_angles_dict.txt", "w") as f:
        force_field = "frcmod.ff14SB"
        ff14sb_bond_angle_dict = extract_bonds_and_angle_info(force_field)
        f.write(pprint.pformat(ff14sb_bond_angle_dict))

    with open("parm10_bonds_angles_dict.txt", "w") as f:
        force_field = "parm10.dat"
        parm10_bond_angle_dict = extract_bonds_and_angle_info(force_field)
        f.write(pprint.pformat(parm10_bond_angle_dict))

    # Add entries from parm10 that are not present in ff14sb
    for subdict in ["bonds", "angles"]:
        sd = ff14sb_bond_angle_dict[subdict]
        sd_parm10 = parm10_bond_angle_dict[subdict]
        for k, v in sd_parm10.items():
            if k not in sd.keys():
                sd[k] = v

    with open("unified_bonds_angles_dict.txt", "w") as f:
        f.write(pprint.pformat(ff14sb_bond_angle_dict))

    with open("full_amino_acid_build_dict.txt", "w") as f:
        full_amino_acid_build_dict = create_full_amino_acid_build_dict(atom_name_dict, ff14sb_bond_angle_dict)
        f.write(pprint.pformat(full_amino_acid_build_dict))



if __name__ == "__main__":
    main()