import openpyxl
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import requests
from xml.etree.ElementTree import fromstring


summaryFile = open('FullSummaryContent', 'r')
lines = summaryFile.readlines()
splittedLines = [line.split('$') for line in lines]
monoUbiquitinSplittedLines = [splittedLine for splittedLine in splittedLines if splittedLine[2] == '1']
monoUbiquitinReceptorsNames = [splittedLine[0] for splittedLine in monoUbiquitinSplittedLines]
ubiquitinBindingResidues = [splittedLine[3] for splittedLine in monoUbiquitinSplittedLines]


def fromBindingResiduesStringToHotOneEncoding(bindingResiduesString):
    hotOneEncoding = np.zeros(75)
    if bindingResiduesString == '\n':
        return hotOneEncoding
    bindingResidues = bindingResiduesString[:-1].split('+')
    bindingResiduesNumbers = ["".join([char for char in bindingResidue if char.isdigit()]) for bindingResidue in
                              bindingResidues]
    bindingResiduesNumbersAsInts = [int(bindingResiduesNumber) for bindingResiduesNumber in bindingResiduesNumbers]
    for num in bindingResiduesNumbersAsInts:
        hotOneEncoding[num - 1] = 1
    return hotOneEncoding


def createHotOneEncodings(ubiquitinBindingResidues):
    ubiquitinBindingEncodings = [None for i in range(len(ubiquitinBindingResidues))]
    for i in range(len(ubiquitinBindingEncodings)):
        ubiquitinBindingEncodings[i] = fromBindingResiduesStringToHotOneEncoding(ubiquitinBindingResidues[i])
    return ubiquitinBindingEncodings


def applyPCA(hotOneEncodings, numerOfComponents):
    pca = PCA(n_components=numerOfComponents)
    pca.fit(hotOneEncodings)
    transformed_data = pca.transform(hotOneEncodings)
    return transformed_data


def Kmeans(transformed_data):
    n_clusters = 5  # Number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(transformed_data)
    labels = kmeans.labels_
    return kmeans, labels


def gaussianMixture(transformed_data):
    n_components = 5  # Number of clusters
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(transformed_data)
    labels = gmm.fit_predict(transformed_data)
    return gmm, labels


def plotResults(transformed_data, labels, gmm):
    cluster_centers = gmm.means_
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, color='red', label='Cluster Centers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA + gmm Clustering')
    plt.show()


print(1)


# plotResults(transformed_data, labels, gmm)

def unpickle(path):
    with open(path, 'rb') as file:
        # Load the pickled object
        myObject = pickle.load(file)
        return myObject


unpropagatedPath = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\model_predictions\\predictions_ubiquitin_ScanNet_PUI_retrained_0108.pkl'
unpropagatedPredictions = unpickle(unpropagatedPath)


def findIndexInTested(receptorName, unpropagatedPredictions):
    try:
        index = unpropagatedPredictions['list_origins'].index(receptorName)
    except:
        return None
    return index


def allValidIndexes(monoUbiquitinReceptorsNames, unpropagatedPredictions):
    validList = [i for i in range(len(monoUbiquitinReceptorsNames)) if
                 findIndexInTested(monoUbiquitinReceptorsNames[i], unpropagatedPredictions) is not None]
    return validList


def calculatePositivesAvaragePredictedProbabilityForReceptor(receptorName, unpropagatedPredictions):
    index = findIndexInTested(receptorName, unpropagatedPredictions)
    predictionsIndex = findIndexInTested(receptorName, unpropagatedPredictions)
    if index is None or predictionsIndex is None:
        return None
    labelsList = unpropagatedPredictions['list_labels'][index]
    predictionsList = unpropagatedPredictions['list_predictions'][predictionsIndex]

    positivesPredictions = [predictionsList[i] for i in range(len(labelsList)) if labelsList[i] == 2 or labelsList[i] == 3]
    averagePositivesPredictions = sum(positivesPredictions) / len(positivesPredictions)
    return averagePositivesPredictions


def calculatePositivesAveragePredictedProbability(monoUbiquitinReceptorsNames, unpropagatedPredictions):
    averagePositivesPredictionsDict = dict()
    for i in range(len(monoUbiquitinReceptorsNames)):
        averagePositivesPredictions = calculatePositivesAvaragePredictedProbabilityForReceptor(
            monoUbiquitinReceptorsNames[i], unpropagatedPredictions)
        if averagePositivesPredictions is not None:
            averagePositivesPredictionsDict[monoUbiquitinReceptorsNames[i]] = averagePositivesPredictions
    return averagePositivesPredictionsDict


def calculateIsCovalentBondForReceptor(receptorName, unpropagatedPredictions, monoUbiquitinSplittedLines,
                                       monoUbiqIndex):
    index = findIndexInTested(receptorName, unpropagatedPredictions)
    predictionsIndex = findIndexInTested(receptorName, unpropagatedPredictions)
    if index is None or predictionsIndex is None:
        return None
    # remove endLine from last residue
    monoUbiquitinSplittedLines[monoUbiqIndex][3] = monoUbiquitinSplittedLines[monoUbiqIndex][3][:-1]
    ubiquitinsBoundedResiduesStrings = monoUbiquitinSplittedLines[monoUbiqIndex][3].split('//')
    CTerminusAminoAcid = 'G75'
    for ubiquitinBoundedResiduesString in ubiquitinsBoundedResiduesStrings:
        ubiquitinBoundedResidues = ubiquitinBoundedResiduesString.split('+')
        if CTerminusAminoAcid in ubiquitinBoundedResidues:
            return True
    return False


def calculateIsCovalentBondForReceptors(monoUbiquitinReceptorsNames):
    isCovalentBondDict = dict()
    for i in range(len(monoUbiquitinReceptorsNames)):
        isCovalentBond = calculateIsCovalentBondForReceptor(monoUbiquitinReceptorsNames[i], unpropagatedPredictions,
                                                            monoUbiquitinSplittedLines,
                                                            i)
        if isCovalentBond is not None:
            isCovalentBondDict[monoUbiquitinReceptorsNames[i]] = isCovalentBond
    return isCovalentBondDict


def calculateClustersForReceptors(monoUbiquitinReceptorsNames, ubiquitinBindingResidues, unpropagatedPredictions):
    receptorsClustersDict = dict()
    hotOneEncodings = createHotOneEncodings(ubiquitinBindingResidues)
    transformed_data = applyPCA(hotOneEncodings, 10)
    gmm, labels = gaussianMixture(transformed_data)
    listOfValidIndexes = allValidIndexes(monoUbiquitinReceptorsNames, unpropagatedPredictions)
    for index in listOfValidIndexes:
        receptorsClustersDict[monoUbiquitinReceptorsNames[index]] = labels[index]
    return receptorsClustersDict

receptorsClustersDict = calculateClustersForReceptors(monoUbiquitinReceptorsNames, ubiquitinBindingResidues,
                                                      unpropagatedPredictions)
isCovalentBondDict = calculateIsCovalentBondForReceptors(monoUbiquitinReceptorsNames)
averagePositivesPredictionsDict = calculatePositivesAveragePredictedProbability(monoUbiquitinReceptorsNames, unpropagatedPredictions)
data = {'ClusterNumber': receptorsClustersDict, 'isCovalent': isCovalentBondDict, 'averagePositivesPredictions': averagePositivesPredictionsDict}
df = pd.DataFrame(data)
df.index.name = 'ReceptorName'
excelFileName = 'orientationAnalysis.xlsx'
df.to_excel(excelFileName, index=True)

# import requests
# import xml.etree.ElementTree as ET
#
# pdb_id = '4hhb.A'
# pdb_mapping_url = 'http://www.rcsb.org/pdb/rest/das/pdb_uniprot_mapping/alignment'
# uniprot_url = 'http://www.uniprot.org/uniprot/{}.xml'
#
# def get_uniprot_accession_id(response_xml):
#     root = ET.fromstring(response_xml)
#     for el in root.findall('.//{http://www.pdb.org/das-pdb-uniprot}accession'):
#         return el.text
#     return None
#
# def get_uniprot_protein_name(uniprot_id):
#     uniprot_response = requests.get(uniprot_url.format(uniprot_id)).text
#     root = ET.fromstring(uniprot_response)
#     full_name_elem = root.find('.//{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}fullName')
#     if full_name_elem is not None:
#         return full_name_elem.text
#     return None
#
# def map_pdb_to_uniprot(pdb_id):
#     pdb_mapping_response = requests.get(pdb_mapping_url, params={'query': pdb_id}).text
#     uniprot_id = get_uniprot_accession_id(pdb_mapping_response)
#     if uniprot_id is not None:
#         uniprot_name = get_uniprot_protein_name(uniprot_id)
#         return {
#             'pdb_id': pdb_id,
#             'uniprot_id': uniprot_id,
#             'uniprot_name': uniprot_name
#         }
#     return None
#
# result = map_pdb_to_uniprot(pdb_id)
# if result is not None:
#     print(result)
# else:
#     print(f"No UniProt information found for PDB ID {pdb_id}")
