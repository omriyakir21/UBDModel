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


def find_pdb_path_with_subword(directory, subword):
    """
    Searches for a .pdb file path containing a specific subword within a given directory.

    Parameters:
    directory (str): The directory path to search within.
    subword (str): The subword to look for in the file paths.

    Returns:
    str: The first .pdb file path containing the subword, or None if no match is found.
    """
    print(f"Searching for {subword} in {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdb") and subword in file:
                return os.path.join(root, file)
    return None


ProteinsToExperiment = ['P53061', 'P07900', 'Q9NQL2', 'Q86VN1', 'Q8IYS0', 'O95793']


def cif_to_pdb(cif_path, pdb_path):
    try:
        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()
        structure = gemmi.make_structure_from_block(block)
        structure.write_pdb(pdb_path)
        return True
    except Exception as e:
        print(f"Problem with {cif_path}: {e}")
        return False


def convert_cif_to_pdb_in_directory(assemblies_dir):
    problematic_count = 0
    with open(os.path.join(assemblies_dir, 'problematic files'), 'w') as problematic_files:
        for filename in os.listdir(assemblies_dir):
            if filename.endswith(".cif"):
                cif_path = os.path.join(assemblies_dir, filename)
                pdb_path = os.path.join(assemblies_dir, filename.replace(".cif", ".pdb"))
                success = cif_to_pdb(cif_path, pdb_path)
                if not success:
                    problematic_count += 1
                    problematic_files.write(filename + '\n')
    print(f"Total problematic proteins: {problematic_count}")


# convert .cif files to pdb files
# convert_cif_to_pdb_in_directory(path.assembliesDir)
# convert_cif_to_pdb_in_directory(os.path.join(path.experimentsDir, 'listOfProteins'))

dali_alligner_object = dali_alligner.DaliAligner()
ref_name = ProteinsToExperiment[0][:4] + 'B'
mov_name = '1cmxA'
print(f'ref_name{ref_name}, mov_name{mov_name}')

ref_path = find_pdb_path_with_subword(path.assembliesDir, mov_name[:4])
mov_path = find_pdb_path_with_subword(os.path.join(path.experimentsDir, 'listOfProteins'), ref_name[:4])
print(f'refPath{ref_path}, movPath{mov_path}')

resultsDict = {}
R, t, rmsd, _ = dali_alligner_object.impose_structure(ref_name, mov_name, ref_path, mov_path, 'temp_dir')
resultsDict['R'] = R
resultsDict['t'] = t
resultsDict['rmsd'] = rmsd

save_as_pickle(resultsDict, path.daliAligments + ref_name + '_' + mov_name + '.pkl')
