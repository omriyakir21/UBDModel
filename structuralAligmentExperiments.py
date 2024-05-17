import pymol2
import os
import shutil
import pickle
import path


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object


def copy_files_to_directory(file_paths, destination_directory):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Copy each file to the destination directory
    for file_path in file_paths:
        # Extract the filename from the file path
        file_name = os.path.basename(file_path)
        # Construct the destination path by joining the destination directory and the filename
        destination_path = os.path.join(destination_directory, file_name)
        # Copy the file to the destination directory
        shutil.copyfile(file_path, destination_path)


# chosenAssemblies = loadPickle(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\chosenAssemblies.pkl')

# os.makedirs('ScanNetDB')
# destination_directory = r'ScanNetDB\Cif'
#
# copy_files_to_directory(chosenAssemblies, destination_directory)

fileName = "/home/iscb/wolfson/omriyakir/UBDModel/ScanNetDB/Cif/8eaz-assembly1.cif"
cifDir = os.path.join(path.ScanNetDB, 'Cif')
PDBDir = os.path.join(path.ScanNetDB, 'PDB')
os.makedirs(PDBDir, exist_ok=True)
proteinName = fileName.split("/")[-1].split(".")[0]
newFileName = os.path.join(PDBDir, proteinName + ".pdb")
with pymol2.PyMOL() as pymol:
    pymol.cmd.load(fileName, proteinName)
    pymol.cmd.save(newFileName, selection=proteinName)


