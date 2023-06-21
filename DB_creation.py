import copy
import os
import pickle

import numpy as np
from Bio import pairwise2
from Bio.PDB import PDBList
from Bio.PDB.MMCIFParser import MMCIFParser
from scipy.sparse.csgraph import connected_components


def read_PDB_names_from_file(path):
    """
    :param path: path of a file containing a coma separated PDB ID's
    :return: list containing all PDB names from the txt file
    """
    PDB_names_list = []
    PDB_text_file = open("ubiquitin_containing_pdb_entries.txt", "r")
    PDB_lines = PDB_text_file.readlines()
    for line in PDB_lines:
        for id in line.split(","):
            PDB_names_list.append(id)
    return PDB_names_list


ubiq_list_path = "ubiquitin_containing_pdb_entries.txt"
# PDB_names_list = read_PDB_names_from_file(ubiq_list_path)
# PDB_names_list_short = copy.deepcopy(PDB_names_list[:20])
# PDB_names_list_short1 = copy.deepcopy(PDB_names_list[:10])
# PDB_names_list_short2 = copy.deepcopy(PDB_names_list[10:20])


# PDB_names_list_short = PDB_names_list[300:450]
# PDB_names_list_short = PDB_names_list[450:600]
# PDB_names_list_short = PDB_names_list[600:750]
# PDB_names_list_short = PDB_names_list[750:]


pdbs_path = 'C:/Users/omriy/pythonProject/ubiq_project/pdbs'
assembliesPath = 'C:/Users/omriy/pythonProject/ubiq_project/assemblies'
pdb1 = PDBList()


def downloadAssemblyFiles(PDB_names_list, pdbListObject, dirPath):
    """
    :param PDB_names_list: list of pdb names
    :param pdbListObject: pdbList Object
    :param dirPath: directory path to add all assemblies
    :return: list of lists- for every pdb name all of the assembly file names
    """
    assemblyPathsList = [[] for i in range(len(PDB_names_list))]
    for i in range(len(PDB_names_list)):
        pdbName = PDB_names_list[i]
        newDirPath = dirPath + "/" + pdbName
        if not os.path.exists(newDirPath):
            os.mkdir(newDirPath)
        assembly_num = 1
        while True:
            assemblyPath = pdbListObject.retrieve_assembly_file(pdbName, assembly_num=assembly_num, pdir=newDirPath,
                                                                file_format="mmCif")
            if os.path.exists(assemblyPath):
                assemblyPathsList[i].append(assemblyPath)
                assembly_num += 1
            else:
                break
    return assemblyPathsList


def downloadAsymetricFiles(PDB_names_list, dirPath):
    fileNames = pdb1.download_pdb_files(pdb_codes=PDB_names_list, overwrite=True, file_format="mmCif",
                                        pdir=dirPath)
    return fileNames


def redownloadFailedAssemblies(PDB_names_list, pdbListObject, dirPath):
    for i in range(len(PDB_names_list)):
        pdbName = PDB_names_list[i]
        newDirPath = dirPath + "/" + pdbName
        num_files = len(os.listdir(newDirPath))
        if num_files == 0:  # failed
            assembly_num = 1
            while True:
                assemblyPath = pdbListObject.retrieve_assembly_file(pdbName, assembly_num=assembly_num, pdir=newDirPath,
                                                                    file_format="mmCif")
                if os.path.exists(assemblyPath):
                    assembly_num += 1
                else:
                    break


# asymetricPaths = downloadAsymetricFiles(PDB_names_list_short,pdbs_path)
# assemblyPathsLists = downloadAssemblyFiles(PDB_names_list_short,pdb1,assembliesPath)
# redownloadFailedAssemblies(PDB_names_list_short, pdb1, assembliesPath)


def createAssemblyPathsLists(assembliesDirPath):
    """
    :param assembliesDirPath: path to the directory containing the assemblies
    :return: assemblyPathsLists where assemblyPathsLists[i] is a list containing all the assembly paths of the i'th pdb structure
    """
    assemblyPathsLists = []
    assemblyNames = []
    for pdbDir in os.listdir(assembliesDirPath):
        assemblyNames.append(pdbDir.lower())
        assemblyPathsList = []
        pdbDirPath = os.path.join(assembliesDirPath, pdbDir)
        for assemblyPath in os.listdir(pdbDirPath):
            assemblyPath = os.path.join(pdbDirPath, assemblyPath)
            assemblyPathsList.append(assemblyPath)
        assemblyPathsLists.append(assemblyPathsList)
    return assemblyPathsLists, assemblyNames


def createAsymetricPathsList(pdbs_path):
    """
    :param pdbs_path: path to the directory containing the asymetric files
    :return:asymetricPaths - a list containing the paths pdb
    """
    asymetricNames = []
    asymetricPaths = []
    for pdb in os.listdir(pdbs_path):
        pdbDirPath = os.path.join(pdbs_path, pdb)
        asymetricPaths.append(pdbDirPath)
        asymetricNames.append(pdb.split('.')[0])
    return asymetricPaths, asymetricNames


# asymetricPaths = createAsymetricPathsList(pdbs_path)
# asymetricPaths = asymetricPaths[:20]
# asymetricPaths1 = asymetricPaths[:10]
# asymetricPaths2 = asymetricPaths[10:20]

# assemblyPathsLists = copy.deepcopy(createAssemblyPathsLists(assembliesPath)[:20])
# assemblyPathsLists1 = copy.deepcopy(createAssemblyPathsLists(assembliesPath)[:10])
# assemblyPathsLists2 = copy.deepcopy(createAssemblyPathsLists(assembliesPath)[10:20])

def orderPathsLists():
    """
    :param asymetricPaths: list of the paths of the downloaded aymetric file of each pdb
    :param assemblyPathsLists: list of lists of the paths of downloaded assemblies for each pdb
    :return: (orderedAsymmetricPaths,orderedAssemblyPathsLists,orderedPDBNameList)
    """
    orderedAsymmetricPaths = []
    orderedAssemblyPathsLists = []
    asymetricPaths, asymetricNames = createAsymetricPathsList(pdbs_path)
    assemblyPathsLists, assemblyNames = createAssemblyPathsLists(assembliesPath)
    orderedPDBNameList = [name for name in asymetricNames if name in assemblyNames]
    for i in range(len(orderedPDBNameList)):
        index = asymetricNames.index(orderedPDBNameList[i])
        orderedAsymmetricPaths.append(asymetricPaths[index])
        index = assemblyNames.index(orderedPDBNameList[i])
        orderedAssemblyPathsLists.append(assemblyPathsLists[index])
    return (orderedAsymmetricPaths, orderedAssemblyPathsLists, orderedPDBNameList)


asymetricPaths, assemblyPathsLists, pdbNamesList = orderPathsLists()


def chooseAssemblys(assembliesPath, PDB_names_list):
    """
    :param assembliesPath: path to the directory containing the assembly directories
    :param PDB_names_list: list of the PDB names
    :return:
    """
    pdbChosenList = []
    for i in range(PDB_names_list):
        pdbDirectoryPath = assembliesPath + '/' + PDB_names_list[i]
        pdbPathList = [f for f in os.listdir(pdbDirectoryPath) if f.endswith('.cif')]
        numOfAssemblies = len(pdbPathList)
        if numOfAssemblies == 0:
            raise Exception("0 assembly files")
        if numOfAssemblies == 1:
            pdbChosenList.append(pdbPathList[0])
        else:
            print(1)  # TODO !!!


parser = MMCIFParser()  # create parser object

# structures1 = [parser.get_structure(PDB_names_list_short1[i], asymetricPaths1[i]) for i in
#               range(len(PDB_names_list_short1))]  # TODO switch between this and the one below
# structures2 = [parser.get_structure(PDB_names_list_short2[i], asymetricPaths2[i]) for i in
#               range(len(PDB_names_list_short2))]  # TODO switch between this and the one below
#

# structures = [parser.get_structure('3B08', 'C:\\Users\\omriy\\pythonProject\\ubiq_project\\pdbs\\3b08.cif') for _ in
#               range(1)]  # TODO switch between this and the one below

threeLetterToSinglelDict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'THR': 'T', 'SER': 'S',
                            'MET': 'M', 'CYS': 'C', 'PRO': 'P', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W', 'HIS': 'H',
                            'LYS': 'K', 'ARG': 'R', 'ASP': 'D', 'GLU': 'E',
                            'ASN': 'N', 'GLN': 'Q'}


def aaOutOfChain(chain):
    """
    :param chain: chain object
    :return: list of aa (not HOH molecule)
    """
    my_list = []
    amino_acids = chain.get_residues()
    for aa in amino_acids:
        name = str(aa.get_resname())
        if name in threeLetterToSinglelDict.keys():  # amino acid and not other molecule
            my_list.append(aa)
    return my_list


def get_str_seq_of_chain(chain):
    """
    :param chain: chain
    :return: Its sequence
    """
    listOfAminoAcids = aaOutOfChain(chain)
    return "".join([threeLetterToSinglelDict[aa.get_resname()] for aa in listOfAminoAcids])


def calculate_identity(seqA, seqB):
    """
    :param seqA: The sequence of amino acid from chain A
    :param seqB: The sequence of amino acid from chain B
    :return: percentage of identity between the sequences
    """
    score = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True, score_only=True)
    # min_len = min(len(seqA), len(seqB))
    # identity = score / min_len
    max_len = max(len(seqA), len(seqB))
    identity = score / max_len
    return identity


def is_ubiq(chain):
    """
    :param chain: a chain
    :return: True iff its a Ubiquitin chain
    """
    seq = get_str_seq_of_chain(chain)
    if len(seq) == 0:  # invalid chain
        return None
    identity_threshold = 0.9
    is_ubiq = identity_threshold < calculate_identity(seq, ubiq_seq)
    return is_ubiq


class Model:
    def classifyModelChains(self):
        """
        :return: The function classifies the model's
        chains to a ubiquitin chains or non ubiquitin chains.
        """
        for chain in self.chains:
            isUbiq = is_ubiq(chain)
            if isUbiq is None:  # chain is invalid
                continue
            if isUbiq:
                self.ubiq_chains.append(chain)
            else:
                self.non_ubiq_chains.append(chain)

    def is_valid_model(self):
        """
        :return: The function returns True iff there is at list one ubiquitin and non ubiquitin chains in the model
        """
        if len(self.ubiq_chains) > 0 and len(self.non_ubiq_chains) > 0:
            return True
        return False

    def __init__(self, model):
        self.chains = model.get_chains()
        self.id = model.id
        self.ubiq_chains = []
        self.non_ubiq_chains = []
        self.classifyModelChains()


class UBD_candidate:

    def createModelsForStructure(self):
        """
        The function add the structure's models to models field (as Model class object)
        :return: None
        """
        for model in self.structure.get_models():
            my_model = Model(model)
            if my_model.is_valid_model():
                self.models.append(my_model)

    def __init__(self, structure):
        self.structure = structure
        self.models = []
        self.createModelsForStructure()


def atomDists(atom1, atoms):
    vector1 = atom1.get_coord()
    vectors = np.array([atom.get_coord() for atom in atoms])
    distances = np.sqrt(((vectors - vector1[np.newaxis]) ** 2).sum(-1))
    return distances


def getAtomsOfChain(chain):
    aminoAcids = aaOutOfChain(chain)
    return getAtomsOfAminoAcids(aminoAcids)


def getAtomsOfAminoAcids(aminoAcids):
    """
    :param aminoAcids: list of a chain's aminoAcids objects
    :return: list of the chain's atoms
    """
    atoms = []
    for aa in aminoAcids:
        atoms += aa.get_atoms()
    return atoms


def calculateDiameter(atoms):
    globalMaxDistance = 0
    for atom in atoms:
        maxDistance = atomDists(atom, atoms).max()
        if maxDistance > globalMaxDistance:
            # print(maxDistance)
            # globalMaxDistance = maxDistance
            globalMaxDistance = copy.copy(maxDistance)
    return globalMaxDistance


def calculateDiameterFromChain(chain):
    atoms = getAtomsOfChain(chain)
    diameter = calculateDiameter(atoms)
    return diameter


projectPath = 'C:/Users/omriy/pythonProject/ubiq_project'
# pdb1.retrieve_pdb_file('3BY4', overwrite=True, file_format="mmCif", pdir=projectPath)
ubiq_path = '3by4.cif'
ubiq_structure = parser.get_structure('3BY4', ubiq_path)
ubiq_chain = ubiq_structure[0]['B']  # uni-prot = "UBIQ-HUMAN"
ubiq_seq = get_str_seq_of_chain(ubiq_chain)
ubiq_amino_acids = aaOutOfChain(ubiq_chain)
ubiq_atoms = getAtomsOfAminoAcids(ubiq_chain)
ubiqDiameter = calculateDiameter(ubiq_atoms)
ubiq_residues_list = [threeLetterToSinglelDict[str(aminoAcid.get_resname())] + str(aminoAcid.get_id()[1]) for aminoAcid
                      in ubiq_amino_acids]

print(ubiq_residues_list)


# UBD_candidates_list1 = [UBD_candidate(structure) for structure in structures1]
# UBD_candidates_list2 = [UBD_candidate(structure) for structure in structures2]

def getCorrespondingUbiqResidues(aaString):
    alignments = pairwise2.align.globalxx(aaString, ubiq_seq)
    # ubiq_residue_list = [ubiq_residus_list[i] for i in range(len(ubiq_residus_list))]
    alignment1 = alignments[0].seqA
    alignment2 = alignments[0].seqB
    # print(alignment1)
    # print(alignment2)
    index1 = 0
    index2 = 0
    correspondingUbiqResidueList = [None for _ in range(len(aaString))]
    for i in range(len(ubiq_seq)):
        if alignment2[i] != '-' and alignment1[i] != '-':
            correspondingUbiqResidueList[index1] = ubiq_residues_list[index2]
        if alignment1[i] != '-':
            index1 += 1
        if alignment2[i] != '-':
            index2 += 1
        if index1 == len(aaString) or index2 == len(ubiq_seq):
            break
    return correspondingUbiqResidueList


def keep_valid_candidates(UBD_candidates_list, PDB_names_list):
    """
    :param UBD_candidates_list: list of UBD_candidates objects
    :return: List with the valid candidates in the list
    """
    assert (len(UBD_candidates_list) == len(PDB_names_list))
    valid_UBD_candidates = []
    valid_PDB_names = []
    for i in range(len(UBD_candidates_list)):
        # for candidate in UBD_candidates_list:
        candidate = UBD_candidates_list[i]
        if len(candidate.models) > 0:
            valid_UBD_candidates.append(candidate)
            valid_PDB_names.append(PDB_names_list[i])
    return valid_UBD_candidates, valid_PDB_names


# valid_UBD_candidates1, valid_PDB_names1 = keep_valid_candidates(UBD_candidates_list1, PDB_names_list_short1)
# valid_UBD_candidates2, valid_PDB_names2 = keep_valid_candidates(UBD_candidates_list2, PDB_names_list_short2)


def keepValidAssemblies(valid_PDB_names, assemblyPathsLists):
    """
    :param valid_PDB_names: names of the valid pdbs (has ubiq and non ubiq)
    :param assemblyPathsLists: where assemblyPathsLists[i] is a list containing all the assembly paths of the i'th pdb structure
    :return: validAssemblyPathsLists which is the assemblies pathsLists of the valid pdbs
    """
    validAssemblyPathsLists = []
    for i in range(len(assemblyPathsLists)):
        assemblyPdbName = assemblyPathsLists[i][0].split("\\")[-2].lower()
        if assemblyPdbName in valid_PDB_names:
            validAssemblyPathsLists.append(assemblyPathsLists[i])
    return validAssemblyPathsLists


# assemblyPathsLists = [['C:\\Users\\omriy\\pythonProject\\ubiq_project\\assemblies\\3B08\\3b08-assembly1.cif',
#                        'C:\\Users\\omriy\\pythonProject\\ubiq_project\\assemblies\\3B08\\3b08-assembly2.cif',
#                        'C:\\Users\\omriy\\pythonProject\\ubiq_project\\assemblies\\3B08\\3b08-assembly3.cif',
#                        'C:\\Users\\omriy\\pythonProject\\ubiq_project\\assemblies\\3B08\\3b08-assembly4.cif']]


# validAssemblyPathsLists1 = keepValidAssemblies(valid_PDB_names1, assemblyPathsLists1)
# validAssemblyPathsLists2 = keepValidAssemblies(valid_PDB_names2, assemblyPathsLists2)


def findLongestNonUbiq(valid_UBD_candidate):
    """
    :param valid_UBD_candidate: UBD_candidate Object
    :return: The sequence of the longest Non ubiq chain in the structure
    """
    model = valid_UBD_candidate.models[0]  # first model
    maxChainLength = 0
    maxChainAminoAcids = None
    for i in range(len(model.non_ubiq_chains)):
        chainAminoAcids = get_str_seq_of_chain(model.non_ubiq_chains[i])
        if len(chainAminoAcids) >= maxChainLength:
            maxChainLength = len(chainAminoAcids)
            maxChainAminoAcids = chainAminoAcids
    return maxChainAminoAcids


def findNumberOfCopiesForSequence(assemblyPath, pdbName, referenceSequence):
    """
    :param assemblyPath: path to assembly file
    :param referenceSequence: string that represents a sequence ( ‘ACDEFGHI’ )
    :return: number of copies the sequence in the assembly structure
    """

    identity_threshold = 0.95
    structure = parser.get_structure(pdbName, assemblyPath)
    numberOfCopies = 0
    structure = UBD_candidate(structure)
    if len(structure.models) == 0:
        return None  # assembly is not valid
    model = structure.models[0]
    toPrintList = []
    if assemblyPath == 'C:/Users/omriy/pythonProject/ubiq_project/assemblies\\1WR6\\1wr6-assembly4.cif':
        toPrintList.append("reference: " + referenceSequence)
        for nonUbiqChain in model.non_ubiq_chains:
            toPrintList.append(nonUbiqChain.full_id[2] + " " + get_str_seq_of_chain(nonUbiqChain))
        # print("\n".join(toPrintList))
    for nonUbiqChain in model.non_ubiq_chains:
        seqToCompare = get_str_seq_of_chain(nonUbiqChain)
        identity = calculate_identity(referenceSequence, seqToCompare)
        if identity_threshold < identity:
            numberOfCopies += 1
    return numberOfCopies


def createEntryDict(index, pdbName, assemblyPathsList, valid_UBD_candidate):
    """
    :param PDB_names_list:
    :param assemblyPathsList:
    :return:
    """
    entryDict = {}
    entryDict['assemblyPathsList'] = assemblyPathsList
    entryDict['index'] = index
    entryDict['entry'] = pdbName
    assemblies = [i + 1 for i in range(len(assemblyPathsList))]
    entryDict['assemblies'] = assemblies
    referenceSequence = findLongestNonUbiq(valid_UBD_candidate)
    entryDict['referenceSequence'] = referenceSequence
    entryDict['referenceCopyNumber'] = []
    for i in range(len(assemblyPathsList)):
        numberofCopies = findNumberOfCopiesForSequence(assemblyPathsList[i], pdbName, referenceSequence)
        if numberofCopies is None:  # not valid assembly
            entryDict['assemblies'].remove(i + 1)
            continue
        entryDict['referenceCopyNumber'].append(numberofCopies)
    return entryDict


def createListOfEntryDicts(valid_UBD_candidates, valid_PDB_names, assemblyPathsLists):
    try:
        assert len(valid_PDB_names) == len(assemblyPathsLists)
        assert len(valid_PDB_names) == len(valid_UBD_candidates)
    except:

        print(len(valid_PDB_names))
        print(len(assemblyPathsLists))
        print(len(valid_UBD_candidates))
        assert (1 == 0)

    entryDictList = []
    for i in range(len(valid_PDB_names)):
        entryDictList.append(createEntryDict(i, valid_PDB_names[i], assemblyPathsLists[i], valid_UBD_candidates[i]))
    return entryDictList


# listOfEntryDicts1 = createListOfEntryDicts(valid_UBD_candidates1, valid_PDB_names1, validAssemblyPathsLists1)
# listOfEntryDicts2 = createListOfEntryDicts(valid_UBD_candidates2, valid_PDB_names2, validAssemblyPathsLists2)


def pickleListOfEntryDicts(listOfEntryDicts, listsSize):
    """
    :param listOfEntryDicts:
    :return:
    """
    addString = "_6"
    pickleDirPath = 'C:\\Users\\omriy\\pythonProject\\ubiq_project\\PickleItems' + str(listsSize) + addString
    for i in range(len(listOfEntryDicts) // listsSize):
        with open(pickleDirPath + "\\listOfEntryDicts" + str(i), "wb") as f:
            # pickle the list to the file
            pickle.dump(listOfEntryDicts[i * listsSize:i * listsSize + listsSize], f)

    with open(pickleDirPath + "\\listOfEntryDicts" + str(len(listOfEntryDicts) // listsSize), "wb") as f:
        # pickle the list to the file
        pickle.dump(listOfEntryDicts[(len(listOfEntryDicts) // listsSize) * listsSize:], f)


def pickleListOfEntryDictsInOne(i, listOfEntryDicts):
    """
    :param listOfEntryDicts:
    :return:
    """
    addString = str(i)
    pickleDirPath = 'C:\\Users\\omriy\\pythonProject\\ubiq_project\\pickle150New'
    with open(pickleDirPath + "\\listOfEntryDicts" + addString, "wb") as f:
        # pickle the list to the file
        pickle.dump(listOfEntryDicts, f)
        print(len(listOfEntryDicts))


# full_tab_for_embed = unpickleTab(1)
# esm_embeddings = full_tab_for_embed['esm_embeddings']

inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14,
           11: 24}  # key=index in Queen value = number of units in multimer

oppositeMap = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 12: 9, 14: 10, 24: 11}


def chooseAssembly(entryAssemblyDict, probabilities, ambiguousFile, notValidFile):
    """
    :param entryAssemblyDict: entryDict of pdb
    :param probabilities: Queen algorithm predictions
    :return: The path of the most likelihood assembly
    """
    referenceCopyNumberString = ' '.join(map(str, entryAssemblyDict['referenceCopyNumber']))
    assembliesString = ' '.join(map(str, entryAssemblyDict['assemblies']))
    probabilitiesString = ' '.join(["(" + str(inv_map[i]) + "," + str(probabilities[i]) + ")" for i in range(12)])
    if len(entryAssemblyDict['referenceCopyNumber']) == 0:  # there werent any valid assemblies
        print("not valid")
        print(entryAssemblyDict['entry'])
        notValidFile.write(
            entryAssemblyDict['entry'] + ": There werent any valid assemblies.\n longestNonUbiqFromAsymetric: " +
            entryAssemblyDict[
                'referenceSequence'] + "\n ,reference CopyNumber is : " + referenceCopyNumberString + "\n")
        return None

    predictions = []
    for val in entryAssemblyDict['referenceCopyNumber']:
        if val in oppositeMap.keys():
            predictions.append(probabilities[oppositeMap[val]])
    if len(predictions) == 0:  # there werent any valid assemblies
        notValidFile.write(
            entryAssemblyDict['entry'] + ": There werent any valid assemblies.\n longestNonUbiqFromAsymetric: " +
            entryAssemblyDict['referenceSequence'] + "\n ,reference CopyNumber is : " + entryAssemblyDict[
                'referenceCopyNumber'] + "\n")
        return None
    if len(predictions) > 1:
        ambiguousFile.write(entryAssemblyDict[
                                'entry'] + ": valid assembly numbers are: " + assembliesString + "\n respective copyNumbers are: "
                            + referenceCopyNumberString + " and respective propbabilities are :" + ' '.join(
            map(str, predictions)) + "\n the total probabilities are: " + probabilitiesString
                            + "\n")

    maxPrediction = max(predictions)
    maxIndex = predictions.index(maxPrediction)
    # print("max index is: ",maxIndex)
    count = entryAssemblyDict['referenceCopyNumber'][maxIndex]
    assemblyNum = entryAssemblyDict['assemblies'][maxIndex]
    # print(assemblyNum)
    assemblyPath = entryAssemblyDict['assemblyPathsList'][maxIndex]
    # print(entryAssemblyDict['assemblyPathsList'])
    # print(assemblyPath)
    return assemblyPath


def chooseAssemblies(listOfEntryDicts, listOfProbabillities, ambiguousFile, notValidFile):
    chosenAssembliesList = []
    for i in range(len(listOfEntryDicts)):
        # if(listOfEntryDicts[i]['entry'] == '4lcd'):
        #     print("found 4lcd")
        #     print(listOfEntryDicts[i]['assemblies'])
        assemblyPath = chooseAssembly(listOfEntryDicts[i], listOfProbabillities[i], ambiguousFile, notValidFile)
        if assemblyPath == None:
            continue
        chosenAssembliesList.append(assemblyPath)
    return chosenAssembliesList


# chosenAssembliesList = chooseAssemblies(listOfEntryDicts, esm_embeddings)
# chosenAssembliesListToFile = '\n'.join(chosenAssembliesList)
# with open("chosenAssemblies.txt", "w") as file:
#     file.write(chosenAssembliesListToFile)  # Write the string to the file


# pickleListOfEntryDicts(listOfEntryDicts, 15)
# pickleListOfEntryDicts(listOfEntryDicts, 50)


# pickleListOfEntryDictsInOne(10, listOfEntryDicts1)
# pickleListOfEntryDictsInOne(20, listOfEntryDicts2)

def fromNamesToPickle(pdbNamesList, assemblyPathsLists, asymetricPaths, start, end):
    PDB_names_list_short = pdbNamesList[start:end]
    assemblyPathsLists_short = assemblyPathsLists[start:end]
    asymetricPaths_short = asymetricPaths[start:end]
    structures = [parser.get_structure(PDB_names_list_short[i], asymetricPaths_short[i]) for i in
                  range(len(PDB_names_list_short))]
    UBD_candidates_list = [UBD_candidate(structure) for structure in structures]

    valid_UBD_candidates, valid_PDB_names = keep_valid_candidates(UBD_candidates_list, PDB_names_list_short)

    validAssemblyPathsLists = keepValidAssemblies(valid_PDB_names, assemblyPathsLists_short)

    listOfEntryDicts = createListOfEntryDicts(valid_UBD_candidates, valid_PDB_names, validAssemblyPathsLists)

    pickleListOfEntryDictsInOne(end, listOfEntryDicts)


# print(len(assemblyPathsLists))
# fromNamesToPickle(pdbNamesList, assemblyPathsLists, asymetricPaths, 0, 150)
# fromNamesToPickle(pdbNamesList, assemblyPathsLists, asymetricPaths, 150, 300)
# fromNamesToPickle(pdbNamesList, assemblyPathsLists, asymetricPaths, 300, 450)
# fromNamesToPickle(pdbNamesList, assemblyPathsLists, asymetricPaths, 450, 600)
# fromNamesToPickle(pdbNamesList, assemblyPathsLists, asymetricPaths, 600, 750)
# fromNamesToPickle(pdbNamesList, assemblyPathsLists, asymetricPaths, 750, 852)

def unpickleTabPkl(path, index):
    with open(path + str(index) + '.pkl', 'rb') as file:
        # Load the pickled object
        myObject = pickle.load(file)
        return myObject


def unpickleTabNoPkl(path, index):
    with open(path + str(index), 'rb') as file:
        # Load the pickled object
        myObject = pickle.load(file)
        return myObject


def fromPickleToChooseAssemblies():
    predictionsList = []
    probabilitiesList = []
    listOfEntryDictsLists = []
    allListsOfEntryDicts = []
    allPredictions = []
    notValidFile = open("notValidAssembliesFileNew.txt", "w")
    ambiguousFile = open('ambiguousFileNew.txt', "w")
    # notValidFile = open("debug_notValidAssembliesFile.txt", "w")
    # ambiguousFile = open('debug_ambiguousFile.txt', "w")
    for i in range(6):
        predictions = unpickleTabPkl('pickleBackNew/pickleBackNew/predictions', i)
        predictions = predictions.tolist()
        predictionsList.append(predictions)
        allPredictions += predictions
        index = (i + 1) * 150 if i != 5 else 852
        listOfEntryDicts = unpickleTabNoPkl('pickle150New/listOfEntryDicts', index)
        allListsOfEntryDicts += listOfEntryDicts
        listOfEntryDictsLists.append(listOfEntryDicts)
    chosenAssemblies = chooseAssemblies(allListsOfEntryDicts, allPredictions, ambiguousFile, notValidFile)
    notValidFile.close()
    ambiguousFile.close()
    print(chosenAssemblies)
    return chosenAssemblies


chosenAssemblies = fromPickleToChooseAssemblies()


def split_list(original_list, num_sublists):
    sublist_size = len(original_list) // num_sublists
    remainder = len(original_list) % num_sublists

    result = []
    index = 0

    for i in range(num_sublists):
        sublist_length = sublist_size + 1 if i < remainder else sublist_size
        sublist = original_list[index:index + sublist_length]
        result.append(sublist)
        index += sublist_length

    return result


chosenAssembliesListOfSublists = split_list(chosenAssemblies, 40)


def atomDist(atom1, atom2):
    """
    :param atom1: atom object
    :param atom2: atom object
    :return: the euclidian distance between the atoms
    """

    vector1 = atom1.get_vector()
    vector2 = atom2.get_vector()
    temp = vector1 - vector2  # subtracting vector
    sum_sq = np.dot(temp, temp)  # sum of the squares
    return np.sqrt(sum_sq)


def getLabelForAA(aa, ubiq_atoms, threshold, diameter=50, diameter_aa=8.):
    """
    :param aa: amino acid object
    :param ubiq_atoms: the ubiquitin atoms
    :return: 1 if there exists an atom that is within 4 Angstrom to a ubiquitin atom else 0
    """
    for atom in aa.get_atoms():
        dists = atomDists(atom, ubiq_atoms)
        if dists.min() < threshold:
            return 1
        elif dists.max() > diameter + diameter_aa + threshold:
            return 0
    return 0


def getLabelsForAminoAcids(aminoAcids, ubiq_atoms, aminoAcidsLabelsList, diameter):
    """
    :param aminoAcids: list of chain's amino acid
    :param ubiq_atoms: ubiquitin atoms
    :param aminoAcidsLabelsList: list of the amino acids labels to be updated
    :return: True iff there is a connection between the chain and the ubiquitin(2 atoms within the threshold distance)
    """
    threshold = 4
    chainUbiqConnection = False
    for i in range(len(aminoAcids)):
        if getLabelForAA(aminoAcids[i], ubiq_atoms, threshold, diameter):
            chainUbiqConnection = True
            aminoAcidsLabelsList[i] = 2
    return chainUbiqConnection


def fillAtrributesAminoAcids(aminoAcids, chain_id, chainAttributesMatrix, aminoAcidsLabelsList):
    """
    :param aminoAcids: list of chain's amino acid
    :param chain_id: The chain's id
    :param aminoAcidsLabelsList:
    :param chainAttributesMatrix:
    The function updates candidateAttributesMatrix such that candidateAttributesMatrix[j] = (chain_id, aa_id , aa_type, aa label)
    """

    for j in range(len(aminoAcids)):
        chainAttributesMatrix[j][0] = chain_id
        chainAttributesMatrix[j][1] = str(aminoAcids[j].get_id()[1])  # set amino acid id
        chainAttributesMatrix[j][2] = threeLetterToSinglelDict[str(aminoAcids[j].get_resname())]  # set amino acid type
        chainAttributesMatrix[j][3] = str(aminoAcidsLabelsList[j])


def checkConnectedAtomsUtil(atomsA, atomsB, n, threshold):
    """
    :param atomsA: list of chain's atoms
    :param atomsB: list of chain's atoms
    :param n: number of atoms to check for chain A
    :param threshold: maximum distance to check between the atoms
    :return: True iff there are at least n pair of atoms (atomA,atomB) within threshold distance from eachother
    """
    cntPairs = 0
    for i in range(len(atomsA)):
        if atomDists(atomsA[i], atomsB).min() < threshold:
            cntPairs += 1
            if cntPairs >= n:
                return True
    return False


def checkConnectedAtoms(aminoAcidsA, aminoAcidsB, n, threshold):
    """
    :param aminoAcidsA: list of chain's amino acid
    :param aminoAcidsB: list of chain's amino acid
    :param n: number of atoms to check for each chain
    :param threshold: maximum distance to check between the atoms
    :return: True iff there are at least n atoms in aminoAcidsA there are within threshold distance from aminoAcidsB
    and there are at least n atoms in aminoAcidsB there are within threshold distance from aminoAcidsA
    """
    atomsA = getAtomsOfAminoAcids(aminoAcidsA)
    atomsB = getAtomsOfAminoAcids(aminoAcidsB)
    if checkConnectedAtomsUtil(atomsA, atomsB, n, threshold):
        return True
    return False


def createAminoAcidLabels(model):
    """
    :param model:
    :return: Tuple : (ubiqNeighbors , nonUbiqNeighbors, modelAttributesMatrix)
    modelAttributesMatrix[i][j] = modelAttributesMatrix[i] = (chain_id, aa_id , aa_type, aa label)
    """
    ubiqNeighbors = [[0 for j in range(len(model.ubiq_chains))] for i in range(len(model.non_ubiq_chains))]
    ubiqChainsAminoAcidLists = [aaOutOfChain(model.ubiq_chains[i]) for i in range(len(model.ubiq_chains))]
    ubiqChainsAtomsLists = [getAtomsOfAminoAcids(ubiqChainsAminoAcidLists[i]) for i in range(len(model.ubiq_chains))]
    nonUbiqChainsAminoAcidLists = [aaOutOfChain(model.non_ubiq_chains[i]) for i in
                                   range(len(model.non_ubiq_chains))]
    modelLabelsMatrix = [[0 for j in range(len(nonUbiqChainsAminoAcidLists[i]))] for i in
                         range(len(model.non_ubiq_chains))]
    modelAttributesMatrix = [[[None, None, None, None] for j in range(len(nonUbiqChainsAminoAcidLists[i]))] for i in
                             range(len(model.non_ubiq_chains))]

    # --------- for each amino acid in the non ubiquitin chain, fill label , type and id stored in  modelAttributesMatrix---------
    # --------- fill ubiqNeighbors matrix: ubiqNeigbors[i][j] == True <-> There is a connection between the i's non ubiquitin chain and the j's ubiquitin chain ---------
    for i in range(len(model.non_ubiq_chains)):  # iterare over the non ubiquitin chains
        for j in range(len(model.ubiq_chains)):  # iterare over the ubiquitin chains
            if getLabelsForAminoAcids(nonUbiqChainsAminoAcidLists[i], ubiqChainsAtomsLists[j],
                                      modelLabelsMatrix[
                                          i],
                                      ubiqDiameter):  # there is a connection between the non ubiquitin chain and the ubiquitin chain
                ubiqNeighbors[i][j] = 1
        chain_id = model.non_ubiq_chains[i].get_id()
        fillAtrributesAminoAcids(nonUbiqChainsAminoAcidLists[i], chain_id, modelAttributesMatrix[i],
                                 modelLabelsMatrix[i])

    # --------- fill nonUbiqNeighbors matrix: ubiqNeigbors[i][j] == True <-> There is a connection between the i's non ubiquitin chain and the j's non ubiquitin chain ---------
    nonUbiqNeighbors = [[0 for j in range(len(model.non_ubiq_chains))] for i in range(len(model.non_ubiq_chains))]
    threshold = 4
    numberOfConnectedAtoms = 10
    for i in range(len(model.non_ubiq_chains)):  # iterare over the non ubiquitin chains
        for j in range(i, len(model.non_ubiq_chains)):  # iterare over the non ubiquitin chains
            if checkConnectedAtoms(nonUbiqChainsAminoAcidLists[i], nonUbiqChainsAminoAcidLists[j],
                                   numberOfConnectedAtoms, threshold):
                nonUbiqNeighbors[i][j] = 1
                nonUbiqNeighbors[j][i] = 1

    return (ubiqNeighbors, nonUbiqNeighbors, modelAttributesMatrix)


def computeConnectedComponents(twoDimList):
    """
    :param twoDimList: A two dimensional list
    :return: Tuple(numComponents , componentsLabels)
    """
    NpNonUbiqNeighbors = np.array(twoDimList)
    return connected_components(csgraph=NpNonUbiqNeighbors, directed=False, return_labels=True)


def createRelatedChainslist(numberOfComponents, labels):
    """
    :param numberOfComponents: number of component = x => 0<=label values<x
    :param labels: labels
    :return: RelatedChainslist: RelatedChainslist[i] = list of all the chain index's which has the label i
    """
    relatedChainsLists = [[] for _ in range(numberOfComponents)]
    for i in range(len(labels)):
        relatedChainsLists[labels[i]].append(i)
    return relatedChainsLists


# def connectivityAlgorithm(ubiqNeighbors,nonUbiqNeighbors):
def connectivityAlgorithm(B, A):
    """
    :param B: (ubiqNeighbors)connectivity matrix (ndarray) of ubiquitin and non-ubiquitin chain in some candidate (dim: numberOf(non-ubiqChains) X numberOf(ubiqchains))
    :param A: (nonUbiqNeighbors)connectivity matrix (ndarray) of non-ubiquitin chains in some candidate (dim: numberOf(non-ubiqChains) X numberOf(non-ubiqChains))
    :return:
    """
    subset = B.sum(1)  # sum of rows (for each non-ubiq chain,number of connections to ubiquitin molecules)
    subset = subset > 0
    connectionIndexList = subset.nonzero()

    A_S = A[subset, :][:, subset]  # sub-graph of non-ubiquitin graph of the chains interacting directly with ubiquitin
    B_S = B[subset, :]  # The corresponding connectivities
    C_S = np.dot(B_S, np.transpose(
        B_S))  # Connectivity matrix of non-ubiquitin chains interacting with at least one same ubiquitin chain
    D_S = np.multiply(A_S,
                      C_S)  # D_S[i][j] == 1 <-> chains i and j in direct contact and interact with at least one same ubiquitin chain
    numComponents, componentsLabels = connected_components(csgraph=D_S, directed=False, return_labels=True)
    return numComponents, componentsLabels, connectionIndexList[0]


def createImerFiles(dirName):
    """
    :return: list of 10 files
    """
    dirName = dirName + '/'
    return [open(dirName + f"Checkchains_{i}_mer.txt", "w") for i in range(1, 25)]


def writeImerToFile(file, modelAttributesMatrix, ithComponentIndexesConverted, candidate, model, index, receptorHeader):
    """
    :param modelAttributesMatrix[i] = a list of the chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    :param ithComponentIndexesConverted- list of model's chain's indexes
    """
    print(file.name)
    lines = []
    lines.append(">" + receptorHeader)
    for i in ithComponentIndexesConverted:
        for aminoAcidAttributes in modelAttributesMatrix[i]:
            lines.append(" ".join(aminoAcidAttributes))
    stringToFile = "\n".join(lines)
    assert (file.write(stringToFile + "\n") > 0)
    logFile = open('logFiles/log' + str(index), "w")
    logFile.write("candidate = " + candidate.structure.get_id() + "\nin file:" + str(file.name))
    logFile.close()


def updateLabelsForChainsUtil(ImerAttributesMatrix, ImerAminoAcids, nonBindingAtoms, nonBindingDiameter):
    threshold = 4
    for i in range(len(ImerAminoAcids)):
        if getLabelForAA(ImerAminoAcids[i], nonBindingAtoms, threshold, nonBindingDiameter):
            if ImerAttributesMatrix[i][3] == '0':  # doesn't bind ubiquitin
                ImerAttributesMatrix[i][3] = '1'
            else:  # bind ubiquitin
                ImerAttributesMatrix[i][3] = '3'


def updateLabelsForChain(ImerAttributesMatrix, nonBindingAttributeMatrix, ImerAminoAcids, nonBindingAminoAcids,
                         ImerAtoms, nonBindingAtoms, nonBindingDiameter):
    """
    :param ImerAttributesMatrix: a list of the first chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    :param nonBindingAttributeMatrix: a list of the second chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    """

    updateLabelsForChainsUtil(ImerAttributesMatrix, ImerAminoAcids, nonBindingAtoms, nonBindingDiameter)


def updateImersLabels(modelAttributesMatrix, ithComponentIndexesConverted, model, nonUbiqDiameters):
    """
    :param modelAttributesMatrix[i]: a list of the chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    :param ithComponentIndexesConverted: list of model's chain's indexes
    """
    aminoAcidsLists = [aaOutOfChain(model.non_ubiq_chains[index]) for index in range(len(model.non_ubiq_chains))]
    atomsLists = [getAtomsOfAminoAcids(aminoAcids) for aminoAcids in aminoAcidsLists]
    for i in ithComponentIndexesConverted:
        for j in range(len(model.non_ubiq_chains)):
            if j not in ithComponentIndexesConverted:  # one is binding non ubiquitin and one is a non-binding-non-ubiquitin
                updateLabelsForChain(modelAttributesMatrix[i], modelAttributesMatrix[j]
                                     , aminoAcidsLists[i],
                                     aminoAcidsLists[j], atomsLists[i],
                                     atomsLists[j],
                                     nonUbiqDiameters[j])


def convertUbiqBindingIndexesList(bindingIndexesList, ubiqCorrespondingList):
    #    print(ubiqCorrespondingList)
    #    print(len(ubiqCorrespondingList))
    #    print(bindingIndexesList)
    #    print(len(bindingIndexesList))
    l = []
    for i in bindingIndexesList:
        try:
            if ubiqCorrespondingList[i] is not None:
                l.append(ubiqCorrespondingList[i])
        except:
            print(i)
            print(ubiqCorrespondingList[i])
            assert (False)

    return l


def createReceptorSummaryUtil(model, ubIndex, nonUbIndex, boundResidueSet, nonUbiqDiameter):
    """
    :param boundResidueSet:
    :param model:
    :param ubIndex:
    :param nonUbIndex:
    :return: a list of the ubiquitin amino acid that bind to the non ubiquitin chain.
    """
    ubAminoAcids = aaOutOfChain(model.ubiq_chains[ubIndex])
    nonUbAminoAcids = aaOutOfChain(model.non_ubiq_chains[nonUbIndex])
    nonUbAtoms = getAtomsOfAminoAcids(nonUbAminoAcids)
    threshold = 4
    # print(ubAminoAcids)
    # print("index: ", nonUbIndex)
    # print("diameter: ", nonUbiqDiameter)
    for i in range(len(ubAminoAcids)):
        if getLabelForAA(ubAminoAcids[i], nonUbAtoms, threshold,nonUbiqDiameter):
            # print(ubAminoAcids[i])
            boundResidueSet.add(i)


def createReceptorSummary(candidate, model, ubiqNeighbors, ithComponentIndexesConverted, ubiqCorrespondingLists,
                          nonUbiqDiameters):
    """
    :param candidate:
    :param model:
    :param ubiqNeighbors: connectivity matrix (ndarray) of ubiquitin and non-ubiquitin chain in some candidate (dim: numberOf(non-ubiqChains) X numberOf(ubiqchains))
    :param ithComponentIndexesConverted: non-ubiquitin chain indexes of the Receptor
    :return: a string of the following format (ReceptorHedear,NumUb,BoundResidueList)
    """
    boundResidueSets = [set() for _ in range(len(model.ubiq_chains))]
    numUb = 0
    receptorHeader = str(candidate.structure.get_id()).lower() + "_" + '+'.join(
        [str(model.id) + "-" + model.non_ubiq_chains[i].get_id() for i in ithComponentIndexesConverted])

    for j in range(len(model.ubiq_chains)):
        bind = False
        for index in ithComponentIndexesConverted:
            if ubiqNeighbors[index][j] == 1:  # The j'th ubiquitin chain is binding with the index's non ubiquitin chain
                # print("receptor ID is ", model.non_ubiq_chains[index].get_id())
                # print("ubiq ID is ", model.ubiq_chains[j].get_id())
                bind = True
                createReceptorSummaryUtil(model, j, index, boundResidueSets[j], nonUbiqDiameters[index])
                # print(boundResidueSets)
        if bind:
            numUb += 1
    boundResidueLists = [list(boundResidueSets[i]) for i in range(len(boundResidueSets))]

    convertedResidueLists = [convertUbiqBindingIndexesList(boundResidueLists[i], ubiqCorrespondingLists[i]) for i in
                             range(len(boundResidueLists))]
    # print(convertedResidueLists)
    # print(convertedResidueLists)
    boundResidueStrings = ["+".join(convertedResidueLists[i]) for i in range(len(convertedResidueLists))]
    # print(boundResidueStrings)
    boundResidueStringsFiltered = [s for s in boundResidueStrings if s != ""]
    return ("//".join(boundResidueStringsFiltered), numUb, receptorHeader)


def createDataBase(tuple):
    """
    :param valid_UBD_candidates: list of UBD_candidates
    :return:
    """

    chosenAssemblies, index = tuple[0], tuple[1]
    indexString = str(index)
    # indexString = '0000'
    assembliesNames = [chosenAssemblies[i].split("\\")[-2].lower() for i in range(len(chosenAssemblies))]
    structures = [parser.get_structure(assembliesNames[i], chosenAssemblies[i]) for i in range(len(chosenAssemblies))]
    # structures = [parser.get_structure(assembliesNames[i], chosenAssemblies[i]) for i in range(len(chosenAssemblies)) if
                #   '3k9o' in assembliesNames[i]]
    UBD_candidates = [UBD_candidate(structure) for structure in structures]
    dirName = "NewBatchNumber" + indexString
    print("\n\n\n creating dir")
    os.mkdir(dirName)
    filesList = createImerFiles(dirName)  # filesList[i] = file containing i-mers if created else None
    summaryLines = []
    summaryFile = open(dirName + '/' + "summaryLog.txt", "w")
    for candidate in UBD_candidates:
        # if candidate.structure.get_id().lower() != '3k9o':
        #     continue
        print(candidate.structure)
        for model in candidate.models:
            print(model)
            nonUbiqDiameters = [calculateDiameterFromChain(NonUbiqChain) for NonUbiqChain in model.non_ubiq_chains]
            # print(nonUbiqDiameters)
            ubiqNeighbors, nonUbiqNeighbors, modelAttributesMatrix = createAminoAcidLabels(model)
            NpNonUbiqNeighbors = np.array(nonUbiqNeighbors)
            NpUbiquitinNeighbors = np.array(ubiqNeighbors)
            numComponents, componentsLabels, connectionIndexList = connectivityAlgorithm(NpUbiquitinNeighbors,
                                                                                         NpNonUbiqNeighbors)
            ubiqCorrespondingLists = [getCorrespondingUbiqResidues(get_str_seq_of_chain(ubiqChain)) for ubiqChain in
                                      model.ubiq_chains]

            for i in range(numComponents):
                ithComponentIndexes = (componentsLabels == i).nonzero()[0]
                ithComponentIndexesConverted = []
                for val in ithComponentIndexes:
                    x = connectionIndexList[val]
                    ithComponentIndexesConverted.append(x)

                updateImersLabels(modelAttributesMatrix, ithComponentIndexesConverted, model, nonUbiqDiameters)

                ubiquitinBindingPatch, numberOfBoundUbiq, receptorHeader = createReceptorSummary(candidate, model,
                                                                                                 ubiqNeighbors,
                                                                                                 ithComponentIndexesConverted,
                                                                                                 ubiqCorrespondingLists,
                                                                                                 nonUbiqDiameters)
                numberOfReceptors = len(ithComponentIndexesConverted)
                summaryLines.append(
                    '$'.join([receptorHeader, str(numberOfReceptors), str(numberOfBoundUbiq), ubiquitinBindingPatch]))
                writeImerToFile(filesList[len(ithComponentIndexesConverted) - 1],
                                modelAttributesMatrix, ithComponentIndexesConverted, candidate, model, index,
                                receptorHeader)

    summaryString = "\n".join(summaryLines)
    assert (summaryFile.write(summaryString) > 0)
    summaryFile.close()
    for file in filesList:
        file.close()


items = [(chosenAssembliesListOfSublists[i], i) for i in range(40)]
# items = (chosenAssemblies,9999)
# items = (chosenAssembliesListOfSublists[0][:3], 99999)
# # print(starmap(createDataBase, items))
# myPool = pool.Pool()
#
# results = myPool.imap(createDataBase, items)
# for result in results:
#     print(result)
# batch7Items = []
# print()
# print(len(items[7][0]))
# size = len(items[7][0])//5
# print(size)
# for i in range(4):
#     batch7Items.append( (items[7][0][i*size:(i+1)*size],8+i))
# batch7Items.append((items[7][0][4*size:],12))
# for batch in batch7Items:
#     print(len(batch[0]))

# createDataBase(items[37])
# createDataBase(items[38])
createDataBase(items[39])
# createDataBase(items[39])
# print(ubiq_seq)
# print(ubiq_amino_acids)
# createDataBase((chosenAssemblies, 2000))
# print(ubiqDiameter)
