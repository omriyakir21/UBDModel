import math
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve
import path
import aggregateScoringMLPUtils as utils
import tensorflow as tf


def createPRPlotFromResults(gridSearchDir, predictions, labels, bestArchitecture):
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
    plt.savefig(os.path.join(gridSearchDir, 'PR plot final model'))
    plt.close()


def createCSVFileFromResults(gridSearchDir, dirName, predictions,
                             allInfoDicts, dictsForTraining):
    dataDictPath = os.path.join(os.path.join(path.GoPath, 'idmapping_2023_12_26.tsv'), 'AllOrganizemsDataDict.pkl')
    yhat_groups = utils.createYhatGroupsFromPredictions(predictions, dictsForTraining, 'test')
    outputPath = os.path.join(gridSearchDir, 'results_final_model' + dirName + '.csv')
    print(outputPath)
    utils.createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, dataDictPath, outputPath, 'test')


def createLogBayesDistributionPlotFromResults(gridSearchDir, predictions, bestArchitecture):
    allLog10Kvalues = [np.log10(utils.KComputation(prediction, 0.05)) if utils.KComputation(prediction, 0.05) != None else math.inf for prediction in predictions]
    plt.hist(allLog10Kvalues)
    plt.title('logKvalues Distribution, architecture = ' + str(bestArchitecture))
    plt.savefig(os.path.join(gridSearchDir, 'logKvalues Distribution final model'))
    plt.close()


dirName = sys.argv[1]
trainingDataDir = os.path.join(path.predictionsToDataSetDir, dirName)
gridSearchDir = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName)
trainingDictsDir = os.path.join(trainingDataDir, 'trainingDicts')
dirPath = os.path.join(gridSearchDir,'finalmodel')
if not os.path.exists(dirPath):
    os.mkdir(dirPath)

allInfoDicts = utils.loadPickle(os.path.join(trainingDictsDir, 'allInfoDicts.pkl'))
dictsForTraining = utils.loadPickle(os.path.join(trainingDictsDir, 'dictsForTraining.pkl'))

directory_name = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName)
if not os.path.exists(directory_name):
    os.mkdir(directory_name)

predictions, labels, bestArchitecture = utils.getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir)
m_a = bestArchitecture[0]
m_b = bestArchitecture[1]
m_c = bestArchitecture[2]
n_layers = bestArchitecture[3]
n_early_stopping_epochs = bestArchitecture[4]
batch_size = bestArchitecture[5]
predictionsDict = {}
modelsList = []
predictions = []
labels = []
for i in range(len(dictsForTraining)):
    tf.keras.utils.clear_session()
    model = utils.buildModelConcatSizeAndNPatchesSameNumberOfLayers(m_a, m_b, m_c, n_layers)
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(curve='PR'), 'accuracy'])
    print(m_a, m_b, m_c, n_layers,
          n_early_stopping_epochs, batch_size, i)
    dictForTraining = dictsForTraining[i]
    allInfoDict = allInfoDicts[i]
    x_train_components_scaled_padded = dictForTraining['x_train_components_scaled_padded']
    x_cv_components_scaled_padded = dictForTraining['x_cv_components_scaled_padded']
    x_test_components_scaled_padded = dictForTraining['x_test_components_scaled_padded']
    x_train_sizes_scaled = dictForTraining['x_train_sizes_scaled']
    x_cv_sizes_scaled = dictForTraining['x_cv_sizes_scaled']
    x_test_sizes_scaled = dictForTraining['x_test_sizes_scaled']
    x_train_n_patches_encoded = dictForTraining['x_train_n_patches_encoded']
    x_cv_n_patches_encoded = dictForTraining['x_cv_n_patches_encoded']
    x_test_n_patches_encoded = dictForTraining['x_test_n_patches_encoded']
    y_train = np.array(dictForTraining['y_train'])
    y_cv = np.array(dictForTraining['y_cv'])
    y_test = np.array(dictForTraining['y_test'])
    class_weights = utils.compute_class_weight('balanced', classes=np.unique(y_train),
                                               y=y_train)
    # Convert class weights to a dictionary
    class_weight = {i: class_weights[i] for i in range(len(class_weights))}
    model.fit(
        [x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded],
        y_train,
        epochs=300,
        verbose=1,
        validation_data=(
            [x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded], y_cv),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                    mode='max',
                                                    patience=n_early_stopping_epochs,
                                                    restore_best_weights=True)],
        batch_size=batch_size

    )

    yhat_test = model.predict(
        [x_test_components_scaled_padded, x_test_sizes_scaled, x_test_n_patches_encoded])
    for j in range(y_test.size):
        uniprot = allInfoDict['x_test'][j][1]
        predictionsDict[uniprot] = (yhat_test[j][0], y_test[j])
        predictions.append(yhat_test[j])
        labels.append(y_test[j])
    tf.saved_model.save(model, os.path.join(dirPath, 'model'+str(i)))


utils.saveAsPickle(predictionsDict, os.path.join(dirPath, 'predictionsDict'))
createPRPlotFromResults(dirPath, predictions, labels, bestArchitecture)
createLogBayesDistributionPlotFromResults(dirPath, predictions, bestArchitecture)
createCSVFileFromResults(dirPath, dirName, predictions, allInfoDicts, dictsForTraining)
