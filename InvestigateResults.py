import heapq
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

scanNetPath = "C:\\Users\\omriy\\UBDAndScanNet\\ScanNet_Ub"
sys.path.append(scanNetPath)
import ScanNet_Ub.utilities.dataset_utils as dataset_utils


class Item:
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority

    def __lt__(self, other):
        # Override the less than operator for proper comparison based on priority
        return self.priority > other.priority


def getTopNMislabledReceptors(trainingResults, n):
    numberOfReceptors = len(trainingResults['list_origins'])
    maxHeap = []
    for i in range(numberOfReceptors):
        for j in range(len(trainingResults["list_origins"][i])):
            if trainingResults["list_labels"][i][j] == 0:
                heapq.heappush(maxHeap, Item(i, trainingResults["all_list_predictions"]['PUI_retrained'][i][j]))

    mislabledReceptorSet = set()
    while len(mislabledReceptorSet) < n:
        largest = heapq.heappop(maxHeap)
        mislabledReceptorSet.add(largest.value)
        print(largest.priority)
    return list(mislabledReceptorSet)


def caluculatePositivesAveragePrediction(labelsList, predictionsList):
    numOfPositiveLabels = sum([int(label) for label in labelsList])
    sumOfPositivePredictions = sum(
        [int(labelsList[i]) * float(np.median(predictionsList[i])) for i in range(len(labelsList))])
    positivesAveragePrediction = sumOfPositivePredictions / numOfPositiveLabels
    return positivesAveragePrediction


def calculateLearningOfExample(labelsList, predictionsList, foldNum):
    deltaPList = np.empty(len(labelsList))
    for i in range(len(labelsList)):
        selfPred = predictionsList[i][foldNum]
        averageNonSelfPredictions = (np.sum(predictionsList[i]) - selfPred) / 4
        deltaP = averageNonSelfPredictions - selfPred
        deltaPList[i] = deltaP
    return np.corrcoef(deltaPList, labelsList)[0][1]


def getTopNUnderfittedReceptors(trainingResults, n):
    numberOfReceptors = len(trainingResults['list_origins'])
    maxHeap = []
    for i in range(numberOfReceptors):
        labelsList = trainingResults['list_labels'][i]
        predictionsList = trainingResults['overfitted_predictions'][i]
        heapq.heappush(maxHeap, Item(i, -1 * caluculatePositivesAveragePrediction(labelsList, predictionsList)))

    underfittedReceptorSet = set()
    while len(underfittedReceptorSet) < n:
        largest = heapq.heappop(maxHeap)
        underfittedReceptorSet.add(largest.value)
        print(largest.priority)
    return list(underfittedReceptorSet)


def plotHistogram(scores, title):
    sns.histplot(scores, kde=True, color='skyblue')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.title(title)
    plt.show()


def plotPositiveReceptoreScoresHistogram(trainingResults):
    numberOfReceptors = len(trainingResults['list_origins'])
    scores = []
    for i in range(numberOfReceptors):
        labelsList = trainingResults['list_labels'][i]
        predictionsList = trainingResults['overfitted_predictions'][i]
        scores.append(caluculatePositivesAveragePrediction(labelsList, predictionsList))
    sns.histplot(scores, kde=True, color='skyblue')
    title = 'positive score of "overfitting" receptors'
    plotHistogram(scores, title)


def unpickle(path):
    with open(path, 'rb') as file:
        # Load the pickled object
        myObject = pickle.load(file)
        return myObject


def plotLearningSuccessOfFolds(trainingResults):
    path = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\receptorsFoldsDict.pkl'
    receptorsFoldsDict = unpickle(path)
    numberOfReceptors = len(trainingResults['list_origins'])
    foldScoresLists = [[] for i in range(5)]
    for i in range(numberOfReceptors):
        if trainingResults['list_origins'][i] not in receptorsFoldsDict:
            continue
        foldNum = receptorsFoldsDict[trainingResults['list_origins'][i]]
        learningScore = calculateLearningOfExample(trainingResults["list_labels"][i],
                                                   trainingResults['overfitted_predictions'][i], foldNum)
        foldScoresLists[foldNum].append(learningScore)
    for i in range(5):
        plotHistogram(foldScoresLists[i], 'learning scores histogram of fold number' + str(i + 1))


with open("C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\analyze_training_results\\training_results_overfitted.pkl",
          'rb') as file:
    trainingResults = pickle.load(file)

# receptorIndexesForMislabeled = getTopNMislabledReceptors(trainingResults, 10)
# 
# print(receptorIndexesForMislabeled)

list_origins, list_sequences, list_resids, list_labels = dataset_utils.read_labels(
    "C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\2606_dataset\\labels_fold1.txt")
# receptorIndexesForUnderfittingPredictions = getTopNUnderfittedReceptors(trainingResults,10)
# print(receptorIndexesForUnderfittingPredictions)
# plotPositiveReceptoreScoresHistogram(trainingResults)
plotLearningSuccessOfFolds(trainingResults)
from predict_bindingsites import write_predictions
from utilities import chimera
from preprocessing import PDBio


def getChimeraAnnotationsOfListOfReceptorsIndexes(receptorIndexes, localDirNameToStorePredictions):
    list_models = [
        'PUI',
        'PUI_noMSA',
        'PUI_old',
        'PUI_old_noMSA'
    ]

    UBD_dir = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel'
    for k in receptorIndexes:
        j = 0
        origin = trainingResults['list_origins'][k]
        resids = trainingResults['list_resids'][k]
        # resids = np.stack([np.zeros(len(resids), dtype=int).astype(str), resids[:, 0], resids[:, 1]], axis=1)

        sequence = trainingResults['list_sequences'][k]
        labels = trainingResults['list_labels'][k]
        predictions = trainingResults['all_list_predictions'][list_models[j]][k]

        predictions_csv_file = os.path.join(UBD_dir, localDirNameToStorePredictions + '/',
                                            'predictions_%s.csv' % origin)
        pdb_file, _ = PDBio.getPDB(origin)
        output_file = os.path.join(UBD_dir, localDirNameToStorePredictions + '/', 'annotated_%s.pdb' % origin)
        write_predictions(predictions_csv_file, resids, sequence, predictions)
        chimera.annotate_pdb_file(pdb_file, predictions_csv_file, output_file, output_script=True, mini=0.0, maxi=0.35,
                                  version='default', field='Binding site probability')

# getChimeraAnnotationsOfListOfReceptorsIndexes(receptorIndexesForUnderfittingPredictions,'analyzeUnderfittingAllFolds')
