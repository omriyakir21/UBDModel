import csv
import pickle
import sys
from itertools import chain
from plistlib import load

import networkx as nx
import numpy as np
import os
import networkx
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
import pandas as pd
import aggregateScoringMLPUtils as utils
from Bio.PDB import MMCIFParser
import path
import proteinLevelDataPartition
import seaborn as sns
from sklearn.metrics import auc

parser = MMCIFParser()
ubdPath = path.mainProjectDir


class SizeDifferentiationException(Exception):
    def __init__(self, uniprotName):
        super().__init__("uniprotName: ", uniprotName, "\n")


class Protein:
    def __init__(self, uniprotName, plddtThreshold):
        self.uniprotName = uniprotName
        self.ubiqPredictions = allPredictions['dict_predictions_ubiquitin'][uniprotName]
        self.nonUbiqPredictions = allPredictions['dict_predictions_interface'][uniprotName]
        self.residues = allPredictions['dict_resids'][uniprotName]
        self.source = self.getSource(allPredictions['dict_sources'][uniprotName])
        self.plddtValues = self.getPlddtValues()
        self.size = None
        self.graph = nx.Graph()
        self.createGraph(plddtThreshold)
        self.connectedComponentsTuples = self.creatConnectedComponentsTuples()

    def getSource(self, source):
        if serverPDBs:
            return source
        if source == 'Human proteome':
            return 'proteome'
        else:
            return source

    def getStructure(self):
        if serverPDBs:
            structurePath = allPredictions['dict_pdb_files'][self.uniprotName]
        else:
            GoPath = path.GoPath
            typePath = os.path.join(GoPath, self.source)
            if self.source == 'proteome':
                structurePath = os.path.join(typePath, 'AF-' + self.uniprotName + '-F1-model_v4.cif')
            else:
                structurePath = os.path.join(typePath, self.uniprotName + '.cif')
        structure = parser.get_structure(self.uniprotName, structurePath)
        return structure

    def getPlddtValues(self):
        structure = self.getStructure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aaOutOfChain(chain)
            return np.array([residues[i].child_list[0].bfactor for i in range(len(residues))])

    def createNodesForGraph(self, residues, plddtThreshold):
        nodes = []
        if len(residues) != len(self.ubiqPredictions):  # need to skip this protein
            raise SizeDifferentiationException(self.uniprotName)
        for i in range(len(residues)):
            plddtVal = residues[i].child_list[0].bfactor
            if plddtVal > plddtThreshold and self.ubiqPredictions[i] > percentile_90:
                nodes.append(i)
        return nodes

    def createEdgesForGraph(self, residues, nodes):
        edges = []
        CAlphaAtoms = [residue["CA"] for residue in residues]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if CAlphaDistance(CAlphaAtoms[i], CAlphaAtoms[j]) < distanceThreshold:
                    edges.append((nodes[i], nodes[j]))
        return edges

    def createGraph(self, plddtThreshold):
        structure = self.getStructure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aaOutOfChain(chain)
            self.size = len(residues)
            nodes = self.createNodesForGraph(residues, plddtThreshold)
            validResidues = [residues[i] for i in nodes]
            edges = self.createEdgesForGraph(validResidues, nodes)
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)

    def creatConnectedComponentsTuples(self):
        tuples = []
        connected_components = list(nx.connected_components(self.graph))
        for componentSet in connected_components:
            averageUbiq, averageNonUbiq, averagePlddt = self.calculateAveragePredictionsForComponent(componentSet)
            length = len(componentSet)
            tuples.append((length, averageUbiq, averageNonUbiq, averagePlddt))
        return tuples

    def calculateAveragePredictionsForComponent(self, indexSet):
        indexes = list(indexSet)
        ubiqPredictions = [self.ubiqPredictions[index] for index in indexes]
        nonUbiqPredictions = [self.nonUbiqPredictions[index] for index in indexes]
        plddtValues = [self.plddtValues[index] for index in indexes]
        assert (len(ubiqPredictions) == len(nonUbiqPredictions) == len(plddtValues))
        averageUbiq = sum(ubiqPredictions) / len(ubiqPredictions)
        averageNonUbiq = sum(nonUbiqPredictions) / len(nonUbiqPredictions)
        averagePlddt = sum(plddtValues) / len(plddtValues)
        return averageUbiq, averageNonUbiq, averagePlddt


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


def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object


def CAlphaDistance(atom1, atom2):
    vector1 = atom1.get_coord()
    vector2 = atom2.get_coord()
    distance = np.sqrt(((vector2[np.newaxis] - vector1[np.newaxis]) ** 2).sum(-1))
    return distance


# # allPredictionsUbiq = {key: allPredictionsUbiq[key] for key in common_keys}
# # dict_resids = {key: allPredictions['dict_resids'][key] for key in common_keys}
# # dict_sequences = {key: allPredictions['dict_sequences'][key] for key in common_keys}
# # dict_sources = {key: allPredictions['dict_sources'][key] for key in common_keys}
# # allPredictions['dict_resids'] = dict_resids
# # allPredictions['dict_sequences'] = dict_sequences
# # allPredictions['dict_sources'] = dict_sources
# # allPredictions['dict_predictions_ubiquitin'] = allPredictionsUbiq
# # saveAsPickle(allPredictions,r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\Predictions\all_predictions_22_3')


def patchesList(allPredictions, i, dirPath, plddtThreshold):
    allKeys = list(allPredictions['dict_resids'].keys())[indexes[i]:indexes[i + 1]]
    proteinObjects = []
    cnt = 0
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
    for key in allKeys:
        print("i= ", i, " cnt = ", cnt, " key = ", key)
        cnt += 1
        try:
            proteinObjects.append(Protein(key, plddtThreshold))
        except SizeDifferentiationException as e:
            print(e)
            continue
        except Exception as e:
            print(e)
            continue
    saveAsPickle(proteinObjects, os.path.join(os.path.join(dirPath, 'proteinObjectsWithEvoluion' + str(i))))


def makeDictWithIntegrationKeys(allPredictions):
    allPredictions2d = loadPickle(os.path.join(ubdPath, os.path.join('Predictions', 'all_predictions_0310.pkl')))
    keys = allPredictions2d['dict_sources'].keys()
    for dictKey in allPredictions.keys():
        allPredictions[dictKey] = {key: allPredictions[dictKey][key] for key in allPredictions[dictKey].keys() if
                                   key in keys}


def pklComponentsAndSource():
    i = sys.argv[1]
    objs = loadPickle(
        os.path.join(ubdPath, os.path.join('newListOfProteinObjects', 'newlistOfProteinObjectsForAggregateFunc' + str(
            i) + '.pkl')))
    tuples = [(obj.source, obj.uniprotName, obj.connectedComponentsTuples) for obj in objs]
    saveAsPickle(tuples, os.path.join(ubdPath, os.path.join('newProteinConnectedComponents',
                                                            'newProteinConnectedComponents' + str(
                                                                i))))


# pklComponentsAndSource()
def repeatingUniprotsToFilter():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(ubdPath, os.path.join('protein_classification', 'uniprotnamecsCSV.csv')))
    # Replace 'your_file.csv' with the actual file path
    # Get unique values from 'proteome' column
    unique_proteome_values = df['proteome'].unique()
    # Find values from 'proteome' that appear in at least one more column
    common_values = set()
    for column in df.columns:
        if column != 'proteome':
            common_values.update(set(unique_proteome_values) & set(df[column]))
    # Display values from 'proteome' that appear in at least one more column
    return list(common_values)


def createLabelsForComponents(allComponents):
    return np.array([0 if component[0] in NegativeSources else 1 for component in allComponents])


def pklLabels(allComponents, dirPath):
    labels = createLabelsForComponents(allComponents)
    labelsDir = os.path.join(dirPath, 'labels')
    os.mkdir(labelsDir)
    saveAsPickle(labels, os.path.join(labelsDir, 'labels'))


def trainKBinDescretizierModel(data, n_bins_parameter):
    est = KBinsDiscretizer(n_bins=n_bins_parameter, encode='ordinal', strategy='quantile', subsample=None)
    est.fit(data)
    return est


def createVectorizedData(kBinModel, allTuplesLists, n_bins_parameter):
    matrixData = [np.zeros([n_bins_parameter, n_bins_parameter]) for _ in range(len(allTuplesLists))]
    for i in range(len(allTuplesLists)):
        for tup in allTuplesLists[i]:
            input_array = np.array(tup).reshape(1, -1)
            integerEncoding = kBinModel.transform(input_array)
            matrixData[i][int(integerEncoding[0][0])][int(integerEncoding[0][1])] += 1
    vectorizedData = np.vstack([matrix.flatten() for matrix in matrixData])
    return vectorizedData


def trainLogisticRegressionModel(X, Y, class_weights=None):
    model = LogisticRegression(class_weight=class_weights)
    model.fit(X, Y)
    return model


def testLogisticRegressionModel(model, X, Y):
    predictions = model.predict(X)
    accuracy = accuracy_score(Y, predictions)
    print(classification_report(Y, predictions))


from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


def plotROC(model, labels, data):
    y_probs = model.predict_proba(data)[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, y_probs)
    auc = roc_auc_score(labels, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def plotPrecisionRecall(y_probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def KComputation(prediction, trainingUbRation):
    val = 1 - prediction
    if val == 0:
        return
    K = ((1 - trainingUbRation) * prediction) / ((trainingUbRation) * (val))
    return K


def predictionFunctionUsingBayesFactorComputation(logisticPrediction, priorUb, trainingUbRatio):
    K = KComputation(logisticPrediction, trainingUbRatio)
    finalPrediction = (K * priorUb) / ((K * priorUb) + (1 - priorUb))
    return finalPrediction


def updateFunction(probabilities, priorUb, trainingUbRatio):
    updatedProbability = predictionFunctionUsingBayesFactorComputation(probabilities[1], priorUb, trainingUbRatio)
    probabilities[1] = updatedProbability
    probabilities[0] = 1 - updatedProbability


def pklComponentsOutOfProteinObjects(dirPath):
    listOfProteinLists = [loadPickle(
        os.path.join(dirPath, 'proteinObjectsWithEvoluion' + str(i) + '.pkl')) for i in
        range(len(indexes) - 1)]
    concatenatedListOfProteins = [protein for sublist in listOfProteinLists for protein in sublist]
    allComponents4d = [(protein.source, protein.uniprotName, protein.connectedComponentsTuples, protein.size,
                        len(protein.connectedComponentsTuples)) for protein in concatenatedListOfProteins]
    componentsDir = os.path.join(dirPath, 'components')
    os.mkdir(componentsDir)
    saveAsPickle(allComponents4d, os.path.join(componentsDir, 'components'))
    return allComponents4d


def getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir):
    totalAucs = loadPickle(os.path.join(gridSearchDir, 'totalAucs.pkl'))
    totalAucs.sort(key=lambda x: -x[1])
    bestArchitecture = totalAucs[0][0]
    m_a = bestArchitecture[0]
    m_b = bestArchitecture[1]
    m_c = bestArchitecture[2]
    layers = bestArchitecture[3]
    predictionsAndLabels = loadPickle(
        os.path.join(gridSearchDir, 'predictions_labels_' + str(layers) + ' ' + str(m_a) + '.pkl'))
    for i in range(len(predictionsAndLabels)):
        if predictionsAndLabels[i][0][1] == m_b and predictionsAndLabels[i][0][2] == m_c:
            predictions = predictionsAndLabels[i][1]
            predictions = np.array([val[0] for val in predictions])
            labels = predictionsAndLabels[i][2]
            break
    return predictions, labels, bestArchitecture


def createCSVFileFromResults(gridSearchDir, trainingDictsDir, dirName):
    predictions, labels, bestArchitecture = getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir)
    allInfoDicts = loadPickle(os.path.join(trainingDictsDir, 'allInfoDicts.pkl'))
    dictsForTraining = loadPickle(os.path.join(trainingDictsDir, 'dictsForTraining.pkl'))
    dataDictPath = os.path.join(os.path.join(path.GoPath, 'idmapping_2023_12_26.tsv'), 'AllOrganizemsDataDict.pkl')
    yhat_groups = utils.createYhatGroupsFromPredictions(predictions, dictsForTraining)
    outputPath = os.path.join(gridSearchDir, 'results_' + dirName + '.csv')
    print(outputPath)
    utils.createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, dataDictPath, outputPath)


def createCombinedCsv(gridSearchDir, dirName, gridSearchDir2, dirName2,plddtThreshold,plddtThreshold2):
    # Read the first CSV file
    df1 = pd.read_csv(os.path.join(gridSearchDir, 'results_' + dirName + '.csv'))

    # Read the second CSV file
    df2 = pd.read_csv(os.path.join(gridSearchDir2, 'results_' + dirName2 + '.csv'))

    # Merge the two dataframes based on common columns
    merged_df = pd.merge(df1, df2, on=["Entry", "type", "Protein Name", "Organism"])

    # Select the desired columns for the new CSV file
    selected_columns = ["Entry", "type", "Protein Name", "Organism", "Inference Prediction 0.05_x", "log10Kvalue_x",
                        "Inference Prediction 0.05_y", "log10Kvalue_y"]

    # Rename the columns to differentiate between the two CSV files
    merged_df.rename(
        columns={"Inference Prediction 0.05 prior_x": "Inference Prediction 0.05_"+str(plddtThreshold), "log10Kvalue_x": "log10Kvalue_"+str(plddtThreshold),
                 "Inference Prediction 0.05 prior_y": "Inference Prediction 0.05_"+str(plddtThreshold2),
                 "log10Kvalue_y": "log10Kvalue_"+str(plddtThreshold2)}, inplace=True)

    # Write the merged dataframe to a new CSV file
    merged_df[selected_columns].to_csv(os.path.join(path.aggregateFunctionMLPDir, "combined_csv_"+str(len(df1['Entry']))+'.csv'), index=False)


def createPRPlotFromResults(gridSearchDir):
    predictions, labels, bestArchitecture = getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir)
    labels = np.array(labels)
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    sorted_indices = np.argsort(recall)
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    aucScore = auc(sorted_recall, sorted_precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve, architecture = ' + str(bestArchitecture) + " auc=" + str(aucScore))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(gridSearchDir, 'PR plot'))
    plt.close()


def createLogBayesDistributionPlotFromResults(gridSearchDir):
    predictions, labels, bestArchitecture = getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir)
    allLog10Kvalues = [np.log10(KComputation(prediction, 0.05)) for prediction in predictions]
    plt.hist(allLog10Kvalues)
    plt.title('logKvalues Distribution, architecture = '+str(bestArchitecture))
    plt.savefig(os.path.join(gridSearchDir, 'logKvalues Distribution'))
    plt.close()


def plotPlddtHistogramForKeys(keys, allPredictions, header):
    avgs = []
    for uniprot in keys:
        print(uniprot)
        structurePath = allPredictions['dict_pdb_files'][uniprot]
        structure = parser.get_structure(uniprot, structurePath)
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aaOutOfChain(chain)
            avg = np.mean(np.array([residues[i].child_list[0].bfactor for i in range(len(residues))]))
            avgs.append(avg)
        sns.histplot(avgs, kde=True)
        plt.title(header)
        plt.show()


def plotPlddtHistogramForPositivieAndProteome(allPredictions):
    keys = allPredictions['dict_sources'].keys()
    positiveKeys = [key for key in keys if
                    allPredictions['dict_sources'][key] in ['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB']][:50]
    proteomeKeys = [key for key in keys if allPredictions['dict_sources'][key] == 'Human proteome'][:50]
    plotPlddtHistogramForKeys(positiveKeys, allPredictions, 'Positives plddt histogram')
    plotPlddtHistogramForKeys(proteomeKeys, allPredictions, 'Proteome plddt histogram')


# !!!!
# JEROME lOOK FROM HERE
serverPDBs = True
NegativeSources = set(
    ['Yeast proteome', 'Human proteome', 'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])
allPredictions = loadPickle(os.path.join(path.ScanNetPredictionsPath, 'all_predictions_0304_MSA_True.pkl'))
allPredictionsUbiq = allPredictions['dict_predictions_ubiquitin']
allPredictionsNonUbiq = allPredictions['dict_predictions_interface']
allPredictionsUbiqFlatten = [value for values_list in allPredictionsUbiq.values() for value in values_list]
percentile_90 = np.percentile(allPredictionsUbiqFlatten, 90)
distanceThreshold = 10
dirName = sys.argv[2]
plddtThreshold = int(sys.argv[3])
trainingDataDir = os.path.join(path.predictionsToDataSetDir, dirName)
gridSearchDir = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName)
indexes = list(range(0, len(allPredictions['dict_resids']) + 1, 1500)) + [len(allPredictions['dict_resids'])]

trainingDictsDir = os.path.join(trainingDataDir, 'trainingDicts')

# plotPlddtHistogramForPositivieAndProteome(allPredictions)

# CREATE PROTEIN OBJECTS, I'M DOING IT IN BATCHES
# patchesList(allPredictions, int(sys.argv[1]), trainingDataDir, plddtThreshold)

# FROM HERE FOLLOWS IN ONE RUN
# PKL ALL THE COMPONENTS TOGETHER AND CREATE LABELS FROM THE PATCHES LIST
# components = pklComponentsOutOfProteinObjects(trainingDataDir)
# labels = pklLabels(components, trainingDataDir)

# CREATE DATA FOR TRAINING (allInfoDicts and dictForTraining)
# componentsDir = os.path.join(trainingDataDir, 'components')
# componentsPath = os.path.join(componentsDir, 'components.pkl')
# labelsDir = os.path.join(trainingDataDir, 'labels')
# labelsPath = os.path.join(labelsDir, 'labels.pkl')
# try:
#     os.mkdir(trainingDictsDir)
# except Exception as e:
#     print(e)
# allInfoDict, dictForTraining = utils.createDataForTraining(componentsPath, labelsPath, trainingDictsDir)

# PARTITION THE DATA
# proteinLevelDataPartition.create_x_y_groups('all_predictions_0304_MSA_True.pkl', trainingDataDir)

# CREATE TRAIN TEST VALIDATION FOR ALL GROUPS
# x_groups = loadPickle(os.path.join(trainingDictsDir, 'x_groups.pkl'))
# y_groups = loadPickle(os.path.join(trainingDictsDir, 'y_groups.pkl'))
# allInfoDicts, dictsForTraining = utils.createTrainValidationTestForAllGroups(x_groups, y_groups, trainingDictsDir)


# CREATING THE CSV FILE
# createCSVFileFromResults(gridSearchDir, trainingDictsDir, dirName)

# PLOT SUMMARY  FILES
createPRPlotFromResults(gridSearchDir)
createLogBayesDistributionPlotFromResults(gridSearchDir)
# THATS IT FROM HERE IT IS NOT RELEVANT

# CREATE COMBINED CSV
# dirName2 = sys.argv[4]
# plddtThreshold2 = sys.argv[5]
# trainingDataDir2 = os.path.join(path.predictionsToDataSetDir, dirName2)
# gridSearchDir2 = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName2)
# createCombinedCsv(gridSearchDir, dirName, gridSearchDir2, dirName2)


# !!!!

# common_values = repeatingUniprotsToFilter()
# # existingUniprotNames = [obj.uniprotName for obj in concatenatedListOfProteins]
# for p in concatenatedListOfProteins:
#     if p.uniprotName in common_values:
#         p.source = 'proteome'
#
#
# # missingUniprotsNames = [key for key in allPredictionsUbiq.keys() if key not in uniprotNames]
#
# allComponents3d = [(protein.source, protein.uniprotName, protein.connectedComponentsTuples, protein.size,
#                     len(protein.connectedComponentsTuples)) for protein in concatenatedListOfProteins]
# # allComponents3dFiltered = [component for component in allComponents3d if component[1] not in common_values]
#
# saveAsPickle(allComponents3d,
#              os.path.join(ubdPath, os.path.join('aggregateFunctionMLP', 'allTuplesListsOfLen3_23_3')))

# allComponents3d = loadPickle(
#     os.path.join(ubdPath, os.path.join('aggregateFunctionMLP', 'allTuplesListsOfLen3_23_3.pkl')))
# labels = loadPickle(
#     os.path.join(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP', 'labels3d_23_3.pkl'))


# KBINS
# # n_bins_parameter = 30  # it will actualli be 30^(number of parameter which is 2 because of len(size,average)
# allComponents3dFiltered = loadPickle(
#     os.path.join(ubdPath, os.path.join('aggregateFunctionMLP', 'allTuplesListsOfLen3.pkl')))
# labels = createLabelsForComponents(allComponents3dFiltered)
# saveAsPickle(labels, os.path.join(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP', 'labels3d'))


# print(sum(labels))
# kBinModel = trainKBinDescretizierModel(concatenated_tuples, n_bins_parameter)
# vectorizedData = createVectorizedData(kBinModel, allTuplesLists, n_bins_parameter)
# logisticRegressionModel = trainLogisticRegressionModel(vectorizedData, labels)
# logisticRegressionModelBalanced = trainLogisticRegressionModel(vectorizedData, labels, 'balanced')
# testLogisticRegressionModel(logisticRegressionModel, vectorizedData, labels)
# testLogisticRegressionModel(logisticRegressionModelBalanced, vectorizedData, labels)
# # plt.matshow(logisticRegressionModel.coef_.reshape([30,30]),vmin=-1.,vmax=1,cmap='jet'); plt.colorbar(); plt.show()
# trainingRatio = sum(labels) / len(allTuplesLists)
# ubProbabillits = np.array([row[1] for row in logisticRegressionModel.predict_proba(vectorizedData)])
# finalOutputsTen = [predictionFunctionUsingBayesFactorComputation(proba, 0.1, trainingRatio) for proba in ubProbabillits]
# finalOutputsFifty = [predictionFunctionUsingBayesFactorComputation(proba, 0.5, trainingRatio) for proba in
#                      ubProbabillits]
# KValues = [KComputation(proba, trainingRatio) for proba in ubProbabillits]
# import csv

def readDataFromUni(fileName):
    data_dict = {}
    # Read the TSV file and populate the dictionary
    with open(fileName,
              'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)  # Get column headers
        for row in tsv_reader:
            key = row[0]  # Use the first column as the key
            row_data = dict(
                zip(header[1:], row[1:]))  # Create a dictionary for the row data (excluding the first column)
            data_dict[key] = row_data
        return data_dict


# data_dict = readDataFromUni(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\GO\idmapping_2023_12_26.tsv\idmapping_2023_12_26.tsv')

def createInfoCsvLogisticRegression(data_dict, predBayes10, predBayes50, KValues):
    myList = []
    for i in range(len(finalOutputsTen)):
        uniDict = data_dict[allComponentsFiltered[i][1]]
        myList.append(
            (uniDict['Entry'], uniDict['Protein names'], uniDict['Organism'], predBayes10[i], predBayes50[i],
             KValues[i]))
    headers = ('Entry', 'Protein Name', 'Organism', 'Bayes Prediction 0.1 prior', 'Bayes Prediction 0.5 prior',
               'K value')
    # Define file path for writing
    file_path = 'InfoFileScoringFunction10And50.csv'
    # Write the data to a TSV file
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        # Write headers
        csv_writer.writerow(headers)

        # Write rows of data
        for row in myList:  # Skip the first row since it contains headers
            csv_writer.writerow(row)


# createInfoCsv(data_dict, finalOutputsTen, finalOutputsFifty, KValues)

def getNBiggestFP(labels, predictions, allComponentsFiltered, N):
    negativeIndexes = [i for i in range(len(labels)) if labels[i] == 0]
    topPredictionsIndexes = sorted(negativeIndexes, key=lambda i: predictions[i], reverse=True)[:N]
    NBiggestFP = [(allComponentsFiltered[topPredictionsIndexes[i]][1], predictions[topPredictionsIndexes[i]]) for i in
                  range(N)]
    return NBiggestFP

# NBiggestFP = getNBiggestFP(labels, finalOutputsFifty, allComponentsFiltered, 10)
