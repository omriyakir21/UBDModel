import os
import sys

import numpy as np
from sklearn.metrics import auc
import path
import aggregateScoringMLPUtils as utils
import tensorflow as tf

dirName = sys.argv[4]
dirPath = os.path.join(path.predictionsToDataSetDir, dirName)
trainingDictsDir = os.path.join(dirPath, 'trainingDicts')

allInfoDicts = utils.loadPickle(os.path.join(trainingDictsDir, 'allInfoDicts.pkl'))
dictsForTraining = utils.loadPickle(os.path.join(trainingDictsDir, 'dictsForTraining.pkl'))

directory_name = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName)
if not os.path.exists(directory_name):
    os.mkdir(directory_name)

allArchitecturesAucs = []
allArchitecturesPredictionsAndLabels = []
totalAucs = []
yhat_groups = []
label_groups = []
y_train_groups = []
n_layers = int(sys.argv[1])
m_a = int(sys.argv[2])
m_b_values = [128, 256, 512]
# m_b_values = [2, 2, 2]
# m_b_values = [2]
# m_c_values = [256, 512]
# m_c_values = [2]
m_c = int(sys.argv[3])
batch_size = 1024
n_early_stopping_epochs = 12
cnt = -1
dictForTraining = dictsForTraining
for m_b in m_b_values:
    all_predictions = []
    all_labels = []
    for i in range(len(dictsForTraining)):
        cnt += 1
        model = utils.buildModelConcatSizeAndNPatchesSameNumberOfLayers(m_a, m_b, m_c, n_layers)
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='binary_crossentropy',
                      metrics=[tf.keras.metrics.AUC(curve='PR'), 'accuracy'])
        architectureAucs = []
        print(m_a, m_b, m_c, n_layers,
              n_early_stopping_epochs, batch_size, i)
        dictForTraining = dictsForTraining[i]
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
        if cnt != 0:
            name = 'val_auc_' + str(cnt)
        else:
            name = 'val_auc'

        model.fit(
            [x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded],
            y_train,
            epochs=300,
            verbose=1,
            validation_data=(
                [x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded], y_cv),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor=name,
                                                        mode='max',
                                                        patience=n_early_stopping_epochs,
                                                        restore_best_weights=True)],
            batch_size=batch_size

        )

        yhat_cv = model.predict(
            [x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded])
        precision, recall, thresholds = utils.precision_recall_curve(y_cv, yhat_cv)
        pr_auc = auc(recall, precision)
        architectureAucs.append(((m_a, m_b, m_c, n_layers,
                                  n_early_stopping_epochs, batch_size, i), pr_auc))
        all_predictions.extend(yhat_cv)
        all_labels.extend(y_cv)
        yhat_groups.append(yhat_cv.reshape(-1))
        y_train_groups.append(y_train)
    allArchitecturesPredictionsAndLabels.append(((m_a, m_b, m_c, n_layers,
                                                  n_early_stopping_epochs, batch_size), all_predictions,
                                                 all_labels))
    allArchitecturesAucs.append(architectureAucs)
    precision, recall, thresholds = utils.precision_recall_curve(all_labels, all_predictions)
    pr_auc = auc(recall, precision)
    totalAucs.append(((m_a, m_b, m_c, n_layers,
                       n_early_stopping_epochs, batch_size), pr_auc))

utils.saveAsPickle(allArchitecturesAucs,
                   os.path.join(directory_name, 'allArchitecturesAucs' + str(n_layers) + " " + str(m_a)))
utils.saveAsPickle(totalAucs, os.path.join(directory_name, 'totalAucs' + str(n_layers) + " " + str(m_a)))
utils.saveAsPickle(allArchitecturesPredictionsAndLabels,
                   os.path.join(directory_name, 'predictions_labels_' + str(n_layers) + " " + str(m_a)))
