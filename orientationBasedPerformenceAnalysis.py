import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from UBDModel import LabelPropagationAlgorithm
from UBDModel import Uniprot_utils

summaryFile = open('FullSummaryContent', 'r')
lines = summaryFile.readlines()
summaryFile.close()
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

    positivesPredictions = [predictionsList[i] for i in range(len(labelsList)) if
                            labelsList[i] == 2 or labelsList[i] == 3]
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


def makeChainDict(chainNames):
    chainDict = dict()
    for chainName in chainNames:
        receptorName = chainName.split('$')[0]
        chainId = chainName.split('$')[1]
        if receptorName in chainDict:
            chainDict[receptorName].append(chainId)
        else:
            chainDict[receptorName] = []
            chainDict[receptorName].append(chainId)
    return chainDict


def lookForClassInStringUtil(dicriptionString, classDictForReceptor):
    lookupStringsDict = {'e1': ['e1', 'activating'], 'e2': ['e2', 'conjugating'], 'e3|e4': ['e3', 'e4', 'ligase'],
                         'deubiquitylase': ['deubiquitylase', 'hydrolase', 'deubiquitinating', 'deubiquitinase',
                                            'protease', 'deubiquitin', 'isopeptidase', 'peptidase']}
    dicriptionStringLower = dicriptionString.lower()
    for key in classDictForReceptor.keys():
        for lookupString in lookupStringsDict[key]:
            if dicriptionStringLower.find(lookupString) != -1:
                classDictForReceptor[key] = True


def findClassForReceptor(pdbName, chainsNames, notFoundTuplesList):
    classDictForReceptor = {'e1': False, 'e2': False, 'e3|e4': False, 'deubiquitylase': False}
    for chainName in chainsNames:
        pdbName4Letters = pdbName[:4]
        print((pdbName4Letters, chainName))
        try:
            _, name, _, _, _ = Uniprot_utils.get_chain_organism(pdbName4Letters, chainName)
            lookForClassInStringUtil(name, classDictForReceptor)
        except:
            print('Exception! ')
            notFoundTuplesList.append((pdbName4Letters, chainName))
    return classDictForReceptor


def findClassForReceptors():
    rootPath = 'C:\\Users\\omriy\\UBDAndScanNet\\'
    _, _, _, chainNames, _, _ = LabelPropagationAlgorithm.splitReceptorsIntoIndividualChains(
        rootPath + '\\UBDModel\\FullPssmContent.txt', rootPath + '\\UBDModel\\normalizedFullASAPssmContent')
    chainDict = makeChainDict(chainNames)
    notFoundTuplesList = []
    e1Dict = dict()
    e2Dict = dict()
    e3e4Dict = dict()
    deubiquitylaseDict = dict()
    for i in range(len(monoUbiquitinReceptorsNames)):
        chainIdsForReceptor = chainDict[monoUbiquitinReceptorsNames[i]]
        classDictForReceptor = findClassForReceptor(monoUbiquitinReceptorsNames[i], chainIdsForReceptor,
                                                    notFoundTuplesList)
        e1Dict[monoUbiquitinReceptorsNames[i]] = classDictForReceptor['e1']
        e2Dict[monoUbiquitinReceptorsNames[i]] = classDictForReceptor['e2']
        e3e4Dict[monoUbiquitinReceptorsNames[i]] = classDictForReceptor['e3|e4']
        deubiquitylaseDict[monoUbiquitinReceptorsNames[i]] = classDictForReceptor['deubiquitylase']
    return e1Dict, e2Dict, e3e4Dict, deubiquitylaseDict, notFoundTuplesList


receptorsClustersDict = calculateClustersForReceptors(monoUbiquitinReceptorsNames, ubiquitinBindingResidues,
                                                      unpropagatedPredictions)
isCovalentBondDict = calculateIsCovalentBondForReceptors(monoUbiquitinReceptorsNames)
# e1Dict, e2Dict, e3e4Dict, deubiquitylaseDict, notFoundTuplesList = findClassForReceptors()
averagePositivesPredictionsDict = calculatePositivesAveragePredictedProbability(monoUbiquitinReceptorsNames, unpropagatedPredictions)

# print(e1Dict)
# print(e2Dict)
# print(e3e4Dict)
# print(deubiquitylaseDict)
# print(notFoundTuplesList)
# classificationList = [e1Dict, e2Dict, e3e4Dict, deubiquitylaseDict, notFoundTuplesList]
# LabelPropagationAlgorithm.saveAsPickle(classificationList, 'classificationList')
classificationList = LabelPropagationAlgorithm.loadPickle('classificationList.pkl')
e1Dict = classificationList[0]
e2Dict = classificationList[1]
e3e4Dict = classificationList[2]
deubiquitylaseDict = classificationList[3]
notFoundTuplesList = classificationList[4]
data = {'ClusterNumber': receptorsClustersDict, 'isCovalent': isCovalentBondDict,
        'averagePositivesPredictions': averagePositivesPredictionsDict, 'e1': e1Dict, 'e2': e2Dict, 'e3|e4': e3e4Dict,
        'deubiquitylase':deubiquitylaseDict}

df = pd.DataFrame(data)
df.index.name = 'ReceptorName'
excelFileName = 'orientationAnalysis.xlsx'
df.to_excel(excelFileName, index=True)