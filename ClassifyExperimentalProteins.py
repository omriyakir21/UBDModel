import dali_alligner
import pickle
import sys
import os
import path
import gemmi


def save_as_pickle(obj, file_path):
    """
    Saves an object to a file using pickle.

    Parameters:
    obj (any): The object to be saved.
    file_path (str): The path to the file where the object should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def find_path_with_subword(directory, subword):
    """
    Searches for a file path containing a specific subword within a given directory.

    Parameters:
    directory (str): The directory path to search within.
    subword (str): The subword to look for in the file paths.

    Returns:
    str: The first file path containing the subword, or None if no match is found.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if subword in file:
                return os.path.join(root, file)
    return None

ProteinsToExperiment = ['P53061', 'P07900', 'Q9NQL2', 'Q86VN1', 'Q8IYS0', 'O95793']

def cif_to_pdb(cif_path, pdb_path):
    # Read the .cif file
    doc = gemmi.cif.read_file(cif_path)
    block = doc.sole_block()
    structure = gemmi.make_structure_from_block(block)

    # Write the structure to a .pdb file
    structure.write_pdb(pdb_path)

# Example usage
cif_path = "/home/iscb/wolfson/omriyakir/UBDModel/assemblies/1cmx-assembly1.cif"
pdb_path = "/home/iscb/wolfson/omriyakir/UBDModel/assemblies/1cmx-assembly1.pdb"
cif_to_pdb(cif_path, pdb_path)

# ref_name = sys.argv[1]
# mov_name = sys.argv[2]
# ref_path = sys.argv[3]
# mov_path = sys.argv[4]
#
# resultsDict = {}
# R, t, rmsd, _ = dali_alligner.impose_structure(ref_name, mov_name, ref_path, mov_path, 'temp_dir')
# resultsDict['R'] = R
# resultsDict['t'] = t
# resultsDict['rmsd'] = rmsd
#
# save_as_pickle(resultsDict, path.daliAligments + ref_name + '_' + mov_name + '.pkl')

