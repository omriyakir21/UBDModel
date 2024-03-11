import os
import pickle
import subprocess
import shutil
import numpy as np
import pandas as pd
import requests
from Bio.PDB import MMCIFParser
import csv


def getUniprotIdsFromGpadFile(path):
    uniprotIds = []
    with open(path, 'r') as gpad_file:
        for line in gpad_file:
            fields = line.strip().split('\t')
            if len(fields) >= 8 and fields[0] == 'UniProtKB':
                uniprotIds.append(fields[1])
    return uniprotIds

def getEvidenceFromGpadFile(path):
    evidenceList = []
    with open(path, 'r') as gpad_file:
        for line in gpad_file:
            fields = line.strip().split('\t')
            if len(fields) >= 8 and fields[0] == 'UniProtKB':
                evidenceList.append(fields[11].split('=')[1])
    return evidenceList


def getUniprotIdsUtil(path):
    uniprot_ids = []
    # Open and read the text file
    with open(path, 'r') as file:
        for line in file:
            # Split each line by tabs or spaces
            parts = line.strip().split('\t')  # You can also use split(' ') if space-separated

            # Check if there are at least two parts in the line
            if len(parts) >= 2:
                # Extract the UniProt ID from the first part
                uniprot_id = parts[0].strip().split(':')[1]

                # Append the UniProt ID to the list
                uniprot_ids.append(uniprot_id)

    return uniprot_ids

def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)

def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object

uniprotNamesDict =dict()
uniprotNamesDict['ubiquitinBinding'] = getUniprotIdsFromGpadFile(
    r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\ubiquitinBinding.gpad')
Evidence = getEvidenceFromGpadFile( r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\ubiquitinBinding.gpad')
# uniprotNamesDict['E1'] = getUniprotIdsUtil(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\E1.txt')
# uniprotNamesDict['E2'] = getUniprotIdsUtil(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\E2.txt')
# uniprotNamesDict['E3'] = getUniprotIdsUtil(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\E3.txt')
# uniprotNamesDict['DUB'] = getUniprotIdsUtil(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\DUB.txt')
# saveAsPickle(uniprotNamesDict, r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\uniprotNamesDict')

def fetchAFModels(uniprotNamesDict, className, i, j):
    uniprotIds = uniprotNamesDict[className]
    apiKey = 'AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'
    cnt = 0
    for uniprotId in uniprotIds[i:j]:
        print('i = ', i, ', j= ', j, ', cnt = ', cnt)
        cnt += 1
        api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{uniprotId}?key={apiKey}'
        # Make a GET request to the AlphaFold API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()[0]
            # Check if .cif file URL is available
            if 'cifUrl' in data:
                # Access the .cif file URL
                cif_url = data['cifUrl']
                # Make a GET request to the .cif file URL to download it
                cif_response = requests.get(cif_url)
                # Check if the .cif file request was successful
                if cif_response.status_code == 200:
                    # Save the .cif file to a local file
                    rootPath = r'C:\Users\omriy\UBDAndScanNet\UBDModel'
                    with open(f'{rootPath}/GO/{className}/{uniprotId}.cif', 'wb') as cif_file:
                        cif_file.write(cif_response.content)
                        print(f".cif file downloaded for {uniprotId}")
                else:
                    print(f"Error downloading .cif file: {cif_response.status_code}")
            else:
                print(f"No .cif file available for {uniprotId}")
        else:
            print(f"Error: {response.status_code} - {response.text}, {api_url}")


# fetchAFModels(uniprotNamesDict,'E2')

# uniprotNamesDict = loadPickle(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\uniprotNamesDict.pkl')
# directory_path =r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\E3'
# files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
# file_count = len(files)
# print(file_count)
# fetchAFModels(uniprotNamesDict, 'E3', 16000, 18582)
# print(len(uniprotNamesDict['DUB']))
# print(len(uniprotNamesDict['ubiquitinBinding']))
# fetchAFModels(uniprotNamesDict, 'ubiquitinBinding', 72000, 831100)
# fetchAFModels(uniprotNamesDict, 'DUB', 2000, 4000)
# fetchAFModels(uniprotNamesDict, 'DUB', 8000, 8800)
# fetchAFModels(uniprotNamesDict, 'E3', 12000, 14000)
# fetchAFModels(uniprotNamesDict, 'E3', 14000, 16000)
# fetchAFModels(uniprotNamesDict, 'E3', 16000, 18582)

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

# pdbFilePath = r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\E1\A0A0A0KE12.cif'
# name = 'A0A0A0KE12'
def isValidAFPrediction(pdbFilePath, name):
    parser = MMCIFParser()  # create parser object
    structure = parser.get_structure(name, pdbFilePath)
    model = structure.child_list[0]
    plddtList = []
    for chain in model:
        residues = aaOutOfChain(chain)
        for residue in residues:
            plddtList.append(residue.child_list[0].bfactor)
    above90Residues = [1 for i in range(len(plddtList)) if plddtList[i] > 90]
    if len(above90Residues) > 100 or (len(above90Residues) / len(plddtList)) > 0.2:
        return True
    return False

# isValidAFPrediction(pdbFilePath,name)
def getAllValidAFPredictionsForType(className):
    rootPath = r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO'
    folderPath = os.path.join(rootPath, className)
    validList = []
    l = len(os.listdir(folderPath))
    cnt = 0
    for name in os.listdir(folderPath):
        cnt += 1
        print('length of folder is: ', l, " cnt = ", cnt)
        filePath = os.path.join(folderPath, name)
        if isValidAFPrediction(filePath, name):
            validList.append(filePath)
    saveAsPickle(validList, os.path.join(rootPath, 'allValidOf' + className))

# getAllValidAFPredictionsForType('ubiquitinBinding')
# f = loadPickle('GO/allValidOfE1.pkl')
# print(1)
def uniprotNamesOfGpadfile(path):
    uniprots = []
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        splittedLine = line.split("\t")
        if splittedLine[0] == 'UniProtKB':
            uniprots.append(splittedLine[1])
    return uniprots


def get_str_seq_of_chain(chain):
    """
    :param chain: chain
    :return: Its sequence
    """
    listOfAminoAcids = aaOutOfChain(chain)
    return "".join([threeLetterToSinglelDict[aa.get_resname()] for aa in listOfAminoAcids])


# getAllValidAFPredictionsForType('E3')
# uniprotNamesDict['ubiquitinBinding'] = uniprotNamesOfGpadfile(
#     r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\ubiquitinBinding.gpad')
# saveAsPickle(uniprotNamesDict, r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\uniprotNamesDictAll')
# f = loadPickle(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\uniprotNamesDictAll.pkl')
# print(1)
# getAllValidAFPredictionsForType('ubiquitinBinding')

def getUniprotSequenceTuplesForType(className):
    rootDir = r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO'
    allValidPaths = loadPickle(os.path.join(rootDir, 'allValidOf' + className + '.pkl'))
    tuples = []
    parser = MMCIFParser()  # create parser object
    for path in allValidPaths:
        print(path)
        name = path.split("\\")[-1].split(".")[-2]
        structure = parser.get_structure(name, path)
        model = structure.child_list[0]
        assert (len(model.child_list) == 1)
        seq = get_str_seq_of_chain(model.child_list[0])
        tuples.append((name, seq))
    return tuples


# fileName = 'nameSeqTuples' + 'proteome'
# saveAsPickle(getUniprotSequenceTuplesForType('proteome'), fileName)


def createPathsTextFileForType(className):
    rootDir = r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO'
    allValidPaths = loadPickle(os.path.join(rootDir, 'allValidOf' + className + '.pkl'))
    filePath = os.path.join(rootDir, className + 'Paths.txt')
    with open(filePath, 'w') as file:
        for path in allValidPaths:
            file.write(path + '\n')


def CreateAllTypesListOfSequences(listOfNameSeqTuples):
    seqList = []
    for list in listOfNameSeqTuples:
        for tup in list:
            seqList.append(tup[1])
    return seqList


# GOPath = '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/GO'
GOPath = r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO'

# allValidOfDUB = loadPickle(os.path.join(GOPath, 'allValidOfDUB.pkl'))
# allValidOfE1 = loadPickle(os.path.join(GOPath, 'allValidOfE1.pkl'))
# allValidOfE2 = loadPickle(os.path.join(GOPath, 'allValidOfE2.pkl'))
# allValidOfE3 = loadPickle(os.path.join(GOPath, 'allValidOfE3.pkl'))
# allValidOfubiquitinBinding = loadPickle(os.path.join(GOPath, 'allValidOfubiquitinBinding.pkl'))
#
# nameSeqTuplesE1 = loadPickle(os.path.join(GOPath, 'nameSeqTuplesE1.pkl'))
# nameSeqTuplesE2 = loadPickle(os.path.join(GOPath, 'nameSeqTuplesE2.pkl'))
# nameSeqTuplesE3 = loadPickle(os.path.join(GOPath, 'nameSeqTuplesE3.pkl'))
# nameSeqTuplesDUB = loadPickle(os.path.join(GOPath, 'nameSeqTuplesDUB.pkl'))
# nameSeqTuplesubiquitinBinding = loadPickle(os.path.join(GOPath, 'nameSeqTuplesubiquitinBinding.pkl'))
# nameSeqTuplesProteome = loadPickle(os.path.join(GOPath, 'nameSeqTuplesproteome.pkl'))


# seqListPositives = CreateAllTypesListOfSequences([nameSeqTuplesE1, nameSeqTuplesE2, nameSeqTuplesE3, nameSeqTuplesDUB,
#                        nameSeqTuplesubiquitinBinding])
# saveAsPickle(seqListPositives, r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\seqListPositives')


# createPathsTextFileForType('ubiquitinBinding')
def cluster_sequences(list_sequences, seqid=0.7, coverage=0.8, covmode='0'):
    path2mmseqs = '/home/omriyakir21/MMseqs2/build/bin//mmseqs'
    path2mmseqstmp = '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/mmseqs2'

    rng = np.random.randint(0, high=int(1e6))
    tmp_input = os.path.join(path2mmseqstmp, 'tmp_input_file_%s.fasta' % rng)
    tmp_output = os.path.join(path2mmseqstmp, 'tmp_output_file_%s' % rng)

    with open(tmp_input, 'w') as f:
        for k, sequence in enumerate(list_sequences):
            f.write('>%s\n' % k)
            f.write('%s\n' % sequence)

    command = ('{mmseqs} easy-cluster {fasta} {result} {tmp} --min-seq-id %s -c %s --cov-mode %s' % (
        seqid, coverage, covmode)).format(mmseqs=path2mmseqs, fasta=tmp_input, result=tmp_output, tmp=path2mmseqstmp)
    subprocess.run(command.split(' '))

    with open(tmp_output + '_rep_seq.fasta', 'r') as f:
        representative_indices = [int(x[1:-1]) for x in f.readlines()[::2]]
    cluster_indices = np.zeros(len(list_sequences), dtype=int)
    table = pd.read_csv(tmp_output + '_cluster.tsv', sep='\t', header=None).to_numpy(dtype=int)
    for i, j in table:
        if i in representative_indices:
            cluster_indices[j] = representative_indices.index(i)
    for file in [tmp_output + '_rep_seq.fasta', tmp_output + '_all_seqs.fasta', tmp_output + '_cluster.tsv']:
        os.remove(file)
    saveAsPickle(cluster_indices, path2mmseqstmp + '/clusterIndices')
    return np.array(cluster_indices), np.array(representative_indices)


# seqListNegatives = [tup[1] for tup in nameSeqTuplesProteome]
#
# ubuntu = False
#
# if ubuntu:
#     Path = '/mnt/c/Usersroot/omriy/UBDAndScanNet'
#
#     cluster_indices, representative_indices = cluster_sequences(seqListNegatives)
#     saveAsPickle(cluster_indices, '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/GO/ClusterIndicesNegatives70')
#     saveAsPickle(representative_indices,
#                  '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/GO/representativeIndicesNegatives70')
# else:
#     rootPath = r'C:\Users\omriy\UBDAndScanNet'
#     seqListPositives = loadPickle(os.path.join(GOPath, 'seqListPositives.pkl'))


# clusterIndexes = loadPickle(rootPath + 'UBDModel/mmseqs2/clusterIndices.pkl')

# ClusterIndicesPositives = loadPickle((os.path.join(GOPath, 'ClusterIndicesPositives95.pkl')))
# representativeIndicesPositives = loadPickle((os.path.join(GOPath, 'representativeIndicesPositives95.pkl')))
# path2mafft = '/usr/bin/mafft'
# indexesDict = dict()
# indexesDict['E1'] = (0, len(nameSeqTuplesE1))
# indexesDict['E2'] = (indexesDict['E1'][1], indexesDict['E1'][1] + len(nameSeqTuplesE2))
# indexesDict['E3'] = (indexesDict['E2'][1], indexesDict['E2'][1] + len(nameSeqTuplesE3))
# indexesDict['DUB'] = (indexesDict['E3'][1], indexesDict['E3'][1] + len(nameSeqTuplesDUB))
# indexesDict['ubiquitinBinding'] = (indexesDict['DUB'][1], indexesDict['DUB'][1] + len(nameSeqTuplesubiquitinBinding))


def creatNameClusterTypeDict(ClusterIndices, nameList, i, j, nameClusterTypeDict, ClassName):
    assert (len(nameList) == j - i)
    for k in range(i, j):
        if nameList[k - i] in nameClusterTypeDict:
            nameClusterTypeDict[nameList[k - i]][0].append(ClassName)
        else:
            nameClusterTypeDict[nameList[k - i]] = ([ClassName], ClusterIndices[k])


def createNameList(nameSeqTuples):
    return [tup[0] for tup in nameSeqTuples]


# ClusterIndicesNegatives = loadPickle(os.path.join(GOPath, 'ClusterIndicesNegatives95.pkl'))
#
# E1NameList = createNameList(nameSeqTuplesE1)
# E2NameList = createNameList(nameSeqTuplesE2)
# E3NameList = createNameList(nameSeqTuplesE3)
# DUBNameList = createNameList(nameSeqTuplesDUB)
# ubiquitinBindingNameList = createNameList(nameSeqTuplesubiquitinBinding)
# proteomeNameList = createNameList(nameSeqTuplesProteome)
#
# nameClusterTypeDictPositives = dict()
# nameClusterTypeDictNegatives = dict()
# creatNameClusterTypeDict(ClusterIndicesPositives, E1NameList, indexesDict['E1'][0], indexesDict['E1'][1],
#                          nameClusterTypeDictPositives, 'E1')
# creatNameClusterTypeDict(ClusterIndicesPositives, E2NameList, indexesDict['E2'][0], indexesDict['E2'][1],
#                          nameClusterTypeDictPositives, 'E2')
# creatNameClusterTypeDict(ClusterIndicesPositives, E3NameList, indexesDict['E3'][0], indexesDict['E3'][1],
#                          nameClusterTypeDictPositives, 'E3')
# creatNameClusterTypeDict(ClusterIndicesPositives, DUBNameList, indexesDict['DUB'][0], indexesDict['DUB'][1],
#                          nameClusterTypeDictPositives, 'DUB')
# creatNameClusterTypeDict(ClusterIndicesPositives, ubiquitinBindingNameList, indexesDict['ubiquitinBinding'][0],
#                          indexesDict['ubiquitinBinding'][1],
#                          nameClusterTypeDictPositives, 'ubiquitinBinding')
# creatNameClusterTypeDict(ClusterIndicesNegatives, proteomeNameList, 0, len(proteomeNameList),
#                          nameClusterTypeDictNegatives, 'Proteome')
# saveAsPickle(nameClusterTypeDictPositives, 'nameClusterTypeDictPositives')
# saveAsPickle(nameClusterTypeDictNegatives, 'nameClusterTypeDictNegatives')


def createTargetsFileForType(className):
    source_directory = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\GO\\' + className
    output_file_path = 'C:\\Users\\omriy\\UBDAndScanNet\\ScanNet_Ub\\tagets' + className + '.txt'
    with open(output_file_path, "w") as output_file:
        # Iterate over files in the source directory
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                # Check if the file has a ".cif" extension
                if file.endswith(".cif"):
                    # Get the absolute path of the source file
                    source_file_path = os.path.join(root, file)
                    # Write the absolute path to the text file
                    output_file.write(source_file_path + "\n")


# createTargetsFileForType('proteome')
def split_file(input_file, output_prefix, lines_per_file=200):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    num_files = total_lines // lines_per_file + (1 if total_lines % lines_per_file != 0 else 0)

    for i in range(num_files):
        start_idx = i * lines_per_file
        end_idx = min((i + 1) * lines_per_file, total_lines)

        output_file = f"{output_prefix}_{i + 1}.txt"

        with open(output_file, 'w') as f_out:
            f_out.writelines(lines[start_idx:end_idx])


# input_file = r'C:\Users\omriy\UBDAndScanNet\ScanNet_Ub\tagetsubiquitinBinding.txt' # Replace with your input file name
# output_prefix = r'C:\Users\omriy\UBDAndScanNet\ScanNet_Ub\targetsubiquitinBinding\targetsubiquitinBinding'
# lines_per_file = 700
#
# split_file(input_file, output_prefix, lines_per_file)


def GetAllValidUniprotIdsOfType(typeName, proteome=False):
    GoPath = r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO'
    allValidList = loadPickle(os.path.join(GoPath, 'allValidOf' + typeName + '.pkl'))
    if proteome:
        return [path.split("\\")[-1].split("-")[1] for path in allValidList]
    return [path.split("\\")[-1][:-4] for path in allValidList]


# uniprotsDict = dict()
# uniprotsDict['E1'] = GetAllValidUniprotIdsOfType('E1')
# uniprotsDict['E2'] = GetAllValidUniprotIdsOfType('E2')
# uniprotsDict['E3'] = GetAllValidUniprotIdsOfType('E3')
# uniprotsDict['DUB'] = GetAllValidUniprotIdsOfType('DUB')
# uniprotsDict['ubiquitinBinding'] = GetAllValidUniprotIdsOfType('ubiquitinBinding')
# uniprotsDict['proteome'] = GetAllValidUniprotIdsOfType('proteome', True)


def write_dict_to_csv(filename, data_dict):
    # Find the maximum length of any list in the dictionary
    max_length = max(len(v) for v in data_dict.values())

    # Fill missing values with an empty string
    for key, value in data_dict.items():
        if len(value) < max_length:
            data_dict[key].extend([''] * (max_length - len(value)))

    # Open the CSV file in write mode
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())

        # Write the header row with column names
        csv_writer.writeheader()

        # Write the data rows
        for i in range(max_length):
            row = {key: data_dict[key][i] for key in data_dict}
            csv_writer.writerow(row)


# write_dict_to_csv(os.path.join(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO', "uniprotnamecsCSV.csv."), uniprotsDict)