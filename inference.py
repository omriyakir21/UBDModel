import numpy as np
from UBDModel import path
import aggragateScoringFileDevelopment as aggragate
import os
import tensorflow as tf
import aggregateScoringMLPUtils as utils

plddtThreshold = 50
modelsDir = os.path.join(path.gridSearchDir, 'finalmodel')
models = [tf.keras.models.load_model(os.path.join(modelsDir, 'model' + str(i) + '.keras')) for i in range(5)]

allInfoDicts = utils.loadPickle(os.path.join(path.trainingDir, 'allInfoDicts.pkl'))

averageUbBindingScaler = utils.loadPickle(os.path.join(modelsDir, 'averageUbBindingScaler.pkl'))
plddtScaler = utils.loadPickle(os.path.join(modelsDir, 'plddtScaler.pkl'))
proteinSizeScaler = utils.loadPickle(os.path.join(modelsDir, 'proteinSizeScaler.pkl'))
sizeComponentScaler = utils.loadPickle(os.path.join(modelsDir, 'sizeComponentScaler.pkl'))
maxNumberOfPatches = 10
uniprot = 'H2QVX7'



def createUniprotSets(allInfoDicts):
    uniprotSets = []
    for i in range(len(allInfoDicts)):
        allInfoDict = allInfoDicts[i]
        uniprotSet = set()
        for j in range(len(allInfoDict['x_test'])):
            uniprotSet.add(allInfoDict['x_test'][j][1])
        uniprotSets.append(uniprotSet)
    utils.saveAsPickle(uniprotSets, os.path.join(path.trainingDir, 'uniprotSets'))


def findModelNumber(uniprot):
    for i in range(len(allInfoDicts)):
        if uniprot in uniprotSets[i]:
            return i


createUniprotSets(allInfoDicts)
uniprotSets = utils.loadPickle(os.path.join(path.trainingDir, 'uniprotSets.pkl'))

def uniprotToPrediction(uniprot):
    if uniprot not in aggragate.allPredictions['dict_resids'].keys():
        raise Exception("uniprot " + str(uniprot) + " not in the DB")
    modelNum = findModelNumber(uniprot)
    trainingUbRatio = np.mean(allInfoDicts[modelNum]['y_train'])
    model = models[findModelNumber(uniprot)]
    protein = aggragate.Protein(uniprot, plddtThreshold)
    tuples = protein.connectedComponentsTuples
    components = np.array([[tup[0], tup[1], tup[2], tup[3]] for tup in tuples])
    componentsLocations = [tup[4] for tup in tuples]
    n_patches = 0
    if len(tuples) != 0:
        # SORT BY UB BINDING PROB
        sorted_indices = tf.argsort(components[:, 1])
        sorted_tensor = tf.gather(components, sorted_indices)
        sortedTensorListed = [sorted_tensor]
        utils.Scale4DUtil(sortedTensorListed, sizeComponentScaler, averageUbBindingScaler, plddtScaler)
        sortedScaledPadded = tf.keras.preprocessing.sequence.pad_sequences(
            sortedTensorListed, padding="post", maxlen=maxNumberOfPatches, dtype='float32')
        n_patches = np.array([np.min([maxNumberOfPatches, sorted_tensor.shape[0]])])
        n_patches_encoded = utils.hotOneEncodeNPatches(n_patches)

    size = protein.size
    sizeScaled = proteinSizeScaler.transform(np.array([size]).reshape(-1, 1))

    yhat = model.predict([sortedScaledPadded, sizeScaled, n_patches_encoded])
    n_tuples = len(tuples)
    significance = [0 for _ in range(n_tuples)]
    # for i in range(n_tuples):
    # if i == n_tuples - 1:
    #     newTuples = tuples[:i]
    # else:
    #     newTuples = tuples[:i] + tuples[i + 1]


# createUniprotSets(allInfoDicts)
# utils.saveScalersForFinalModel(modelsDir, os.path.join(trainingDir, 'allInfoDict.pkl'))
# if uniprot not in aggragate.allPredictions['dict_resids'].keys():
#     raise Exception("uniprot " + str(uniprot) + " not in the DB")
modelNum = findModelNumber(uniprot)
modelIndex = findModelNumber(uniprot)
model = models[modelIndex]
trainingUbRatio = np.mean(allInfoDicts[modelIndex]['y_train'])
protein = aggragate.Protein(uniprot, plddtThreshold)
tuples = protein.connectedComponentsTuples
components = np.array([[tup[0], tup[1], tup[2], tup[3]] for tup in tuples])
componentsLocations = [tup[4] for tup in tuples]
n_patches = 0
if len(tuples) != 0:
    # SORT BY UB BINDING PROB
    sorted_indices = tf.argsort(components[:, 1])
    sorted_tensor = tf.gather(components, sorted_indices)
    sortedTensorListed = [sorted_tensor]
    utils.Scale4DUtil(sortedTensorListed, sizeComponentScaler, averageUbBindingScaler, plddtScaler)
    sortedScaledPadded = tf.keras.preprocessing.sequence.pad_sequences(
        sortedTensorListed, padding="post", maxlen=maxNumberOfPatches, dtype='float32')
    n_patches = np.array([np.min([maxNumberOfPatches, sorted_tensor.shape[0]])])
    n_patches_encoded = utils.hotOneEncodeNPatches(n_patches)

size = protein.size
sizeScaled = proteinSizeScaler.transform(np.array([size]).reshape(-1, 1))

yhat = model.predict([sortedScaledPadded, sizeScaled, n_patches_encoded])
utils.KComputation(yhat[0], 0.4)
for i, tup in enumerate(allInfoDicts[2]['x_train']):
    if tup[1] == uniprot:
        index = i

n_tuples = len(tuples)
significance = [0 for _ in range(n_tuples)]
# for i in range(n_tuples):
#     if i == n_tuples - 1:
#         newTuples = tuples[:i]
#     else:
#         newTuples = tuples[:i] + tuples[i + 1]
#
