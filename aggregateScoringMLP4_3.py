import os
import sys

import numpy as np
from sklearn.metrics import auc
import path
import aggregateScoringMLPUtils as utils
import tensorflow as tf

allInfoDicts = utils.loadPickle(
    os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining23_3', 'allInfoDicts.pkl')))
dictsForTraining = utils.loadPickle(
    os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining23_3', 'dictsForTraining.pkl')))
directory_name = os.path.join(path.aggregateFunctionMLPDir, 'gridSearch31_3WithEvolutionTrainingAccuracyStopping')
os.mkdir(directory_name)

allArchitecturesAucs = []
totalAucs = []
all_predictions = []
all_labels = []
yhat_groups = []
label_groups = []
y_train_groups = []
n_layers = int(sys.argv[1])
# n_layers = 4
m_a = int(sys.argv[2])
# m_a = 256
m_values = [128, 256]
# m_values = [4, 8, 16, 32, 64, 128, 256]
batch_size = 1024
n_early_stopping_epochs = 5

for m_b in m_values:
    for m_c in m_values:
        model = utils.buildModelConcatSizeAndNPatchesSameNumberOfLayers(m_a, m_b, m_c, n_layers)
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
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
            architectureAucs.append(((m_a, m_b, m_c, n_layers,
                                      n_early_stopping_epochs, batch_size, i), pr_auc))
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

utils.saveAsPickle(allArchitecturesAucs, os.path.join(directory_name, 'allArchitecturesAucs' + str(n_layers) + " " + str(m_a)))
utils.saveAsPickle(totalAucs, os.path.join(directory_name, 'totalAucs' + str(n_layers) + " " + str(m_a)))

# dictForTraining = dictsForTraining
# for m_b in m_values:
#     for m_c in m_values:
#         model = utils.buildModelConcatSizeAndNPatchesSameNumberOfLayers(m_a, m_b, m_c, n_layers)
#         # Compile the model
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#                       loss='binary_crossentropy',
#                       metrics=['accuracy'])
#         architectureAucs = []
#         x_train_components_scaled_padded = dictForTraining['x_train_components_scaled_padded']
#         x_cv_components_scaled_padded = dictForTraining['x_cv_components_scaled_padded']
#         x_test_components_scaled_padded = dictForTraining['x_test_components_scaled_padded']
#         x_train_sizes_scaled = dictForTraining['x_train_sizes_scaled']
#         x_cv_sizes_scaled = dictForTraining['x_cv_sizes_scaled']
#         x_test_sizes_scaled = dictForTraining['x_test_sizes_scaled']
#         x_train_n_patches_encoded = dictForTraining['x_train_n_patches_encoded']
#         x_cv_n_patches_encoded = dictForTraining['x_cv_n_patches_encoded']
#         x_test_n_patches_encoded = dictForTraining['x_test_n_patches_encoded']
#         y_train = np.array(dictForTraining['y_train'])
#         y_cv = np.array(dictForTraining['y_cv'])
#         y_test = np.array(dictForTraining['y_test'])
#         class_weights = utils.compute_class_weight('balanced', classes=np.unique(y_train),
#                                                    y=y_train)
#         # Convert class weights to a dictionary
#         class_weight = {i: class_weights[i] for i in range(len(class_weights))}
#
#         # Print the summary of the model
#         model.fit(
#             [x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded],
#             y_train,
#             epochs=300,
#             verbose=1,
#             validation_data=(
#                 [x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded], y_cv),
#             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy',
#                                                         patience=n_early_stopping_epochs)],
#             batch_size=batch_size,
#             class_weight=class_weight
#
#         )
#
#         yhat_cv = model.predict(
#             [x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded])
#         precision, recall, thresholds = utils.precision_recall_curve(y_cv, yhat_cv)
#         pr_auc = auc(recall, precision)
#         print(pr_auc)


