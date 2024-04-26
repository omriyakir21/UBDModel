import pickle
import os


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
    return object


def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def concatAllFilesOfName(dirPath, name):
    concatenated_lists = []
    for filename in os.listdir(dirPath):
        if filename.startswith(name):
            filePath = os.path.join(dirPath, filename)
            # Load the pickle file
            data = loadPickle(filePath)
            # Concatenate the lists
            concatenated_lists.extend(data)
    return concatenated_lists


def concatAllHalfPredictions(dirPath):
    predictionsDict = {}
    for filename in os.listdir(dirPath):
        if ' m_c' in filename:
            prefix = filename.split(' m_c')[0]
            predictions = loadPickle(os.path.join(dirPath, filename))
            if prefix not in predictionsDict:
                predictionsDict[prefix] = predictions
            else:
                predictionsDict[prefix] += predictions
    for key in predictionsDict:
        filename = key.split('m_a')[0]
        saveAsPickle(predictionsDict[key], os.path.join(dirPath, filename))


dirPath = '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/MLP_MSA_val_AUC_stoppage_with_evolution_85_plddt_all_organizems_15_4/'
name = 'Allarchite'

concatenated_lists = concatAllFilesOfName(dirPath, name)
saveAsPickle(concatenated_lists, os.path.join(dirPath, 'allArchitectureAucs'))
name = 'totalA'
concatenated_lists = concatAllFilesOfName(dirPath, name)
saveAsPickle(concatenated_lists, os.path.join(dirPath, 'totalAucs'))

concatAllHalfPredictions(dirPath)
