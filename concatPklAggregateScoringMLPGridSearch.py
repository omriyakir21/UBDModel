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
    dictsDict = []
    for filename in os.listdir(dirPath):
        if ' m_c' in filename:
            prefix = filename.split(' m_c')[0]
            predictions = loadPickle(os.path.join(dirPath, filename))
            if prefix not in dictsDict:
                dictsDict[prefix] = predictions
            else:
                dictsDict[prefix].update(predictions)
    for key in dictsDict:
        filename = key.split('m_a')[0]
        saveAsPickle(dictsDict[key], os.path.join(dirPath, filename))


dirPath = '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/gridSearch6_3/'
name = 'Allarchite'

concatenated_lists = concatAllFilesOfName(dirPath, name)
saveAsPickle(concatenated_lists, os.path.join(dirPath, 'allArchitectureAucs'))
