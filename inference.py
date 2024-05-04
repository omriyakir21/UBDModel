import numpy as np

import aggragateScoringFileDevelopment as aggragate
import os
import tensorflow as tf
import aggregateScoringMLPUtils as utils
import tensorflow.keras.layers as layers
import pandas as pd



plddtThreshold = 50
gridSearchDir = ('/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP'
                 '/MLP_MSA_val_AUC_stoppage_with_evolution_50_plddt_all_organizems_15_4')
modelsDir = os.path.join(gridSearchDir, 'finalmodel')
models = [tf.keras.models.load_model(os.path.join(modelsDir, 'model' + str(i)+'.keras')) for i in range(5)]
trainingDir = ('/home/iscb/wolfson/omriyakir/UBDModel/predictionsToDataSet/with_evolution_50_plddt_all_organizems_15_4'
               '/trainingDicts/')
allInfoDicts = utils.loadPickle(os.path.join(trainingDir, 'allInfoDicts.pkl'))
dictsForTraining = utils.loadPickle(os.path.join(trainingDir, 'dictsForTraining.pkl'))
uniprotSets = utils.loadPickle(os.path.join(trainingDir, 'uniprotSets.pkl'))
averageUbBindingScaler = utils.loadPickle(os.path.join(modelsDir,'averageUbBindingScaler.pkl'))
plddtScaler = utils.loadPickle(os.path.join(modelsDir,'plddtScaler.pkl'))
proteinSizeScaler = utils.loadPickle(os.path.join(modelsDir,'proteinSizeScaler.pkl'))
sizeComponentScaler = utils.loadPickle(os.path.join(modelsDir,'sizeComponentScaler.pkl'))
maxNumberOfPatches = 10
uniprot = 'A0A7K5BL29'

def strPatchFromListIndexes(residues,listLocations):
    myList = [aggragate.threeLetterToSinglelDict[residues[index].resname]+str(residues[index].id[1]) for index in listLocations]
    strPatch = ','.join(myList)
    return strPatch


def createStrPatchesAndSignificance10(significance,sortedLocationsCutted,uniprot,protein):
    structure = protein.getStructure()
    model = structure.child_list[0]
    assert (len(model) == 1)
    for chain in model:
        residues = aggragate.aaOutOfChain(chain)
    strPatches = [None for _ in range(10)]
    significance10 = [None for _ in range(10)]
    for i in range(len(significance)):
        strPatch = strPatchFromListIndexes(residues,sortedLocationsCutted[significance[i][0]])
        strPatches[i] = strPatch
        significance10[i] = significance[i][1]
    return strPatches,significance10

def createUniprotSets(allInfoDicts):
    uniprotSets = []
    for i in range(len(allInfoDicts)):
        allInfoDict = allInfoDicts[i]
        uniprotSet = set()
        for j in range(len(allInfoDict['x_test'])):
            uniprotSet.add(allInfoDict['x_test'][j][1])
        uniprotSets.append(uniprotSet)
    utils.saveAsPickle(uniprotSets, os.path.join(trainingDir, 'uniprotSets'))


def findModelNumber(uniprot):
    for i in range(len(allInfoDicts)):
        if uniprot in uniprotSets[i]:
            return i

def sortLocations(componentsLocations,sorted_indices):
    sortedLocations = []
    for i in range(len(componentsLocations)):
        sortedLocations.append(componentsLocations[sorted_indices[i]])
    return sortedLocations

def sortBestPatchesFromUniprot(uniprot):
    if uniprot not in aggragate.allPredictions['dict_resids'].keys():
        raise Exception("uniprot " + str(uniprot) + " not in the DB")
    modelNum = findModelNumber(uniprot)
    modelIndex = findModelNumber(uniprot)
    model = models[modelIndex]
    trainingUbRatio = np.mean(allInfoDicts[modelIndex]['y_train'])
    protein = aggragate.Protein(uniprot, plddtThreshold)
    tuples = protein.connectedComponentsTuples
    components = np.array([[tup[0], tup[1], tup[2], tup[3]] for tup in tuples])
    componentsLocations = [tup[4] for tup in tuples]
    n_patches = 0
    if len(tuples) == 0:
        return [None for i in range(10)],[None for i in range(10)]
    # SORT BY UB BINDING PROB
    sorted_indices = tf.argsort(components[:, 1])
    sorted_tensor = tf.gather(components, sorted_indices)
    sortedTensorListed = [sorted_tensor]
    utils.Scale4DUtil(sortedTensorListed, sizeComponentScaler, averageUbBindingScaler, plddtScaler)
    sortedScaledPadded = tf.keras.preprocessing.sequence.pad_sequences(
        sortedTensorListed, padding="post", maxlen=maxNumberOfPatches, dtype='float32')
    n_patches = np.array([np.min([maxNumberOfPatches, sorted_tensor.shape[0]])])
    n_patches_encoded = utils.hotOneEncodeNPatches(n_patches)
    sortedLocations = sortLocations(componentsLocations, sorted_indices)
    sortedLocationsCutted = sortedLocations[:n_patches[0]]

    size = protein.size
    sizeScaled = proteinSizeScaler.transform(np.array([size]).reshape(-1, 1))

    yhat = model.predict([sortedScaledPadded, sizeScaled, n_patches_encoded])
    KValue = utils.KComputation(yhat[0], trainingUbRatio)
    inferencePrediction = utils.predictionFunctionUsingBayesFactorComputation(0.05, KValue)

    significance = [None for _ in range(n_patches[0])]
    for i in range(n_patches[0]):
        newComponents = np.delete(sortedScaledPadded, i, axis=1)
        location = sortedLocationsCutted[i]
        new_n_patches_encoded = utils.hotOneEncodeNPatches(n_patches - 1)
        newYhat = model.predict([newComponents, sizeScaled, new_n_patches_encoded])
        newKValue = utils.KComputation(newYhat[0], trainingUbRatio)
        newInferencePrediction = utils.predictionFunctionUsingBayesFactorComputation(0.05, newKValue)
        significance[i] = (i, inferencePrediction - newInferencePrediction)

    significance.sort(key=lambda x: -x[1])

    strPatches,significance10 = createStrPatchesAndSignificance10(significance,sortedLocationsCutted,uniprot,protein)
    return strPatches,significance10



# Human = df[df['type'] == 'Human proteome']
# sortedHuman = df.sort_values(by='Inference Prediction 0.05 prior', ascending=False)
# sortedHuman100 = sortedHuman.head(100)

# strPatches,significance10 = sortBestPatchesFromUniprot(uniprot)
def createCsvForType(type,numOfType):
    finalReslutsPath = os.path.join(modelsDir,'results_final_modelwith_evolution_50_plddt_all_organizems_15_4.csv')
    df = pd.read_csv(finalReslutsPath)
    typeDf = df[df['type'] == type]
    sortedDf = typeDf.sort_values(by='Inference Prediction 0.05 prior', ascending=False)
    sortedCutted = sortedDf.head(numOfType)
    uniprots = sortedCutted['Entry'].to_list()
    strPatchesLists = [[] for i in range(10)]
    significanceLists = [[] for i in range(10)]
    for i in range(numOfType):
        print(uniprots[i])
        strPatches, significance10 = sortBestPatchesFromUniprot(uniprots[i])
        for j in range(10):
            strPatchesLists[j].append(strPatches[j])
            significanceLists[j].append(significance10[j])

    for i in range(10):
        sortedCutted['Patch'+str(i)] = strPatchesLists[i]
        sortedCutted['Reduced Probability'+str(i)] = significanceLists[i]
    sortedCutted.to_csv(os.path.join(modelsDir,type+'.csv'), index=False)
    return sortedCutted


for type in list(aggragate.NegativeSources):
    createCsvForType(type,1000)