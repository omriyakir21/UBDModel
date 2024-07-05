import logging
import re
import shutil
import subprocess
import tempfile
# from objects import Protein 
import numpy as np
import os
import sys
import path
import pickle


def save_as_pickle(obj, file_path):
    """
    Saves an object to a file using pickle.

    Parameters:
    obj (any): The object to be saved.
    file_path (str): The path to the file where the object should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


logging.getLogger('matplotlib').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

ref_protein = sys.argv[1]
mov_protein = sys.argv[2]


class DaliAligner():
    HOME_PATH = "/home/iscb/wolfson/omriyakir"
    DAT_PATH = os.path.join(HOME_PATH, "DaliLite.v5/DAT")
    IMPORT_PATH = os.path.join(HOME_PATH, "DaliLite.v5/bin/import.pl")
    DALI_PATH = os.path.join(HOME_PATH, "DaliLite.v5/bin/dali.pl")
    DALI_ALIGMENTS_PATH = path.daliAligments

    def __init__(self) -> None:
        self.name = "DaliAligner"
        self._check_paths()

    def _check_paths(self):
        assert os.path.exists(self.HOME_PATH), f"{self.HOME_PATH} dosn't exist, can't use Dali"
        assert os.path.exists(self.DAT_PATH), f"{self.DAT_PATH} dosn't exist, can't use Dali"
        assert os.path.exists(self.IMPORT_PATH), f"{self.IMPORT_PATH} dosn't exist, can't use Dali"
        assert os.path.exists(self.DALI_PATH), f"{self.DALI_PATH} dosn't exist, can't use Dali"

    @staticmethod
    def extract_matrices_combined(file_path: str) -> tuple[np.ndarray, float, float]:
        matrices = []
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == 3:
                    measure_line = line
                if line.startswith("-matrix"):
                    match = re.findall(r'-?\d+\.\d+', line)
                    if match:
                        matrices.append([float(num) for num in match])

            if len(matrices) == 3:
                file.seek(0)
                rmsd = measure_line.split('  ')[3]
                z = measure_line.split(' ')[6]
                return np.array(matrices), rmsd, z
        return None, None, None

    def impose_structure(self, ref_protein, mov_protein, temp_dir: str) -> tuple[
        list[np.ndarray], list[np.ndarray]]:

        # mov_path = os.path.join("ligand_alligner", ligand_dir, 'pdb' + mov_protein._pdb_name + '.ent')
        mov_path = os.path.join('..', mov_protein + '_non_ligand.ent')
        ref_path = os.path.join('..', ref_protein + '_non_ligand.ent')
        # ref_path = os.path.join("ligand_alligner", ligand_dir, 'pdb' + ref_name + '.ent')
        try:
            temp_dir = tempfile.mkdtemp(prefix=os.path.join(path.daliAligments, "temp_dir", temp_dir + '/'))
            # os.makedirs(temp_dir, exist_ok=True)
            os.chdir(temp_dir)
            import_1 = subprocess.run(
                [self.IMPORT_PATH, '--pdbfile', mov_path, '--pdbid', mov_protein, '--dat', self.DAT_PATH],
                capture_output=True, text=True, check=True)
            # print(import_1)
            import_2 = subprocess.run(
                [self.IMPORT_PATH, '--pdbfile', ref_path, '--pdbid', ref_protein, '--dat', self.DAT_PATH],
                capture_output=True, text=True, check=True)
            allign_log = subprocess.run([self.DALI_PATH, '--cd1', ref_protein, '--cd2', mov_protein,
                                         '--dat1', self.DAT_PATH, '--dat2', self.DAT_PATH, '--title',
                                         "output options", '--outfmt', "summary,alignments,equivalences,transrot",
                                         "--clean"
                                         ], capture_output=True, text=True, check=True)

            matrix, rmsd, _ = DaliAligner.extract_matrices_combined(f'{ref_protein}.txt')
            try:
                os.remove(ref_protein + '.dssp')
                os.remove(mov_protein + '.dssp')
                shutil.rmtree(temp_dir)

            except OSError:
                pass

            os.chdir('/home/iscb/wolfson/omriyakir/ligand_alligner')
            if matrix is not None:
                R = np.linalg.inv(matrix[:, :3])
                t = matrix[:, 3]
                return [R], [t], float(rmsd), []
            else:
                return [], [], [], []

        except Exception as e:
            print(e)
            os.chdir('/home/iscb/wolfson/omriyakir/ligand_alligner')
            return [], [], [], []


resultsDict = {}
R, t, rmsd, _ = DaliAligner().impose_structure(ref_protein, mov_protein, 'temp_dir')
resultsDict['R'] = R
resultsDict['t'] = t
resultsDict['rmsd'] = rmsd

save_as_pickle(resultsDict, path.daliAligments + ref_protein + '_' + mov_protein + '.pkl')
