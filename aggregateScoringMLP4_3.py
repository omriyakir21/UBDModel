import os
import sys

import numpy as np
from sklearn.metrics import auc
from UBDModel import path
import aggregateScoringMLPUtils as utils
import tensorflow as tf

allInfoDicts = utils.loadPickle(
    os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining2902', 'allInfoDicts.pkl')))
dictsForTraining = utils.loadPickle(
    os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining2902', 'dictsForTraining.pkl')))

allArchitecturesAucs = []
totalAucs = []
all_predictions = []
all_labels = []
yhat_groups = []
label_groups = []
y_train_groups = []
n_layers = int(sys.argv[1])
m_values = [4, 8, 16, 32, 64, 128, 256]
batch_size = 1024
n_early_stopping_epochs = 5
for m_a in m_values:
    for m_b in m_values:
        for m_c in m_values:
            model = utils.buildModelConcatSizeAndNPatchesSameNumberOfLayers(m_a, m_b, m_c, n_layers)
            # Compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            architectureAucs = []
            for i in range(len(dictsForTraining)):
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

                # Print the summary of the model
                model.fit(
                    [x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded],
                    y_train,
                    epochs=300,
                    verbose=1,
                    validation_data=(
                        [x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded], y_cv),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                                                                patience=n_early_stopping_epochs)],
                    batch_size=batch_size,
                    class_weight=class_weight

                )

                yhat_cv = model.predict(
                    [x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded])
                precision, recall, thresholds = utils.precision_recall_curve(y_cv, yhat_cv)
                pr_auc = auc(recall, precision)
                architectureAucs.append((i, pr_auc))
                all_predictions.extend(yhat_cv)
                all_labels.extend(y_cv)
                yhat_groups.append(yhat_cv.reshape(-1))
                y_train_groups.append(y_train)

            allArchitecturesAucs.append(architectureAucs)
            precision, recall, thresholds = utils.precision_recall_curve(all_labels, all_predictions)
            pr_auc = auc(recall, precision)
            totalAucs.append(((m_a, m_b, m_c, n_layers,
                               n_early_stopping_epochs, batch_size),
                              pr_auc))
            print(((m_a, m_b, m_c, n_layers,
                    n_early_stopping_epochs, batch_size),
                   pr_auc))

directory_name = os.path.join(path.aggregateFunctionMLPDir, 'gridSearch4_3')
utils.saveAsPickle(allArchitecturesAucs, os.path.join(path.aggregateFunctionMLPDir,
                                                      os.path.join('gridSearch4_3', 'allArchitecturesAucs' + n_layers)))
utils.saveAsPickle(totalAucs, os.path.join(path.aggregateFunctionMLPDir, os.path.join('gridSearch', 'totalAucs' + n_layers)))
