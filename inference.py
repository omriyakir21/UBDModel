import aggragateScoringFileDevelopment as aggragate
import os
import tensorflow as tf
import aggregateScoringMLPUtils as utils


plddtThreshold = 50
gridSearchDir = ('/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP'
                 '/MLP_MSA_val_AUC_stoppage_with_evolution_50_plddt_all_organizems_15_4')
modelsDir = os.path.join(gridSearchDir, 'finalmodel')
# models = [tf.keras.models.load_model(os.path.join(modelsDir, 'model' + str(i))) for i in range(5)]
trainingDir = ('/home/iscb/wolfson/omriyakir/UBDModel/predictionsToDataSet/with_evolution_50_plddt_all_organizems_15_4'
               '/trainingDicts/')
allInfoDicts = utils.loadPickle(os.path.join(trainingDir, 'allInfoDicts.pkl'))
# uniprotSets = utils.loadPickle(os.path.join(trainingDir, 'uniprotSets.pkl'))

def createUniprotSets(allInfoDicts):
    uniprotSets = []
    for i in range(len(allInfoDicts)):
        allInfoDict = allInfoDicts[i]
        uniprotSet = set()
        for j in range(len(allInfoDict['x_test'])):
            uniprotSet.add(allInfoDict['x_test'][j][1])
        uniprotSets.append(uniprotSet)
    utils.saveAsPickle(uniprotSets, os.path.join(trainingDir, 'uniprotSets'))


# def findModelNumber(uniprot):
#     for i in range(allInfoDicts):
#         if uniprot in uniprotSets[i]:
#             return i


def uniprotToPrediction(uniprot):
    if uniprot not in aggragate.allPredictions:
        raise Exception("uniprot " + str(uniprot) + " not in the DB")
    # model = models[findModelNumber(uniprot)]
    protein = aggragate.Protein(uniprot, plddtThreshold)
    tuples = protein.connectedComponentsTuples
    n_tuples = len(tuples)
    significance = [0 for _ in range(n_tuples)]
    for i in range(n_tuples):
        if i == n_tuples - 1:
            newTuples = tuples[:i]
        else:
            newTuples = tuples[:i] + tuples[i + 1]


createUniprotSets(allInfoDicts)
utils.saveScalersForFinalModel(modelsDir, os.path.join(trainingDir, 'allInfoDict.pkl'))