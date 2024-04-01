import os

import numpy as np
from sklearn.metrics import auc
import path
import aggregateScoringMLPUtils as utils
import tensorflow as tf


# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)





# x_groups = loadPickle(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\x_groups.pkl')
# y_groups = loadPickle(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\y_groups.pkl')
# allInfoDicts, dictsForTraining = createTrainValidationTestForAllGroups(x_groups, y_groups)



# all_predictions = []
# all_labels = []
# yhat_groups = []
# label_groups = []
# y_train_groups = []
# for i in range(len(dictsForTraining)):
#     dictForTraining = dictsForTraining[i]
#     x_train_components_scaled_padded = dictForTraining['x_train_components_scaled_padded']
#     x_cv_components_scaled_padded = dictForTraining['x_cv_components_scaled_padded']
#     x_test_components_scaled_padded = dictForTraining['x_test_components_scaled_padded']
#     x_train_sizes_scaled = dictForTraining['x_train_sizes_scaled']
#     x_cv_sizes_scaled = dictForTraining['x_cv_sizes_scaled']
#     x_test_sizes_scaled = dictForTraining['x_test_sizes_scaled']
#     x_train_n_patches_encoded = dictForTraining['x_train_n_patches_encoded']
#     x_cv_n_patches_encoded = dictForTraining['x_cv_n_patches_encoded']
#     x_test_n_patches_encoded = dictForTraining['x_test_n_patches_encoded']
#     y_train = np.array(dictForTraining['y_train'])
#     y_cv = np.array(dictForTraining['y_cv'])
#     y_test = np.array(dictForTraining['y_test'])
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     # Convert class weights to a dictionary
#     class_weight = {i: class_weights[i] for i in range(len(class_weights))}
#
#     # Define the input shape
#     input_shape = (maxNumberOfPatches, 3)
#     input_data = tf.keras.Input(shape=input_shape, name='sensor_input')
#     extra_value = tf.keras.Input(shape=(1,), name='extra_value_input')
#     hot_encoded_value = tf.keras.Input(shape=(maxNumberOfPatches + 1,), name='hot_encoded_value_input')
#     masked_input = tf.keras.layers.Masking(mask_value=0.0)(input_data)
#     dense_output = tf.keras.layers.Dense(64, activation='relu')(masked_input)
#     dense_output = tf.keras.layers.Dense(32, activation='relu')(dense_output)
#     global_pooling_output = GlobalSumPooling(data_format='channels_last')(dense_output)
#     concatenated_output = tf.keras.layers.Concatenate()([global_pooling_output, extra_value, hot_encoded_value])
#     dense_output = tf.keras.layers.Dense(64, activation='relu')(concatenated_output)
#     dense_output = tf.keras.layers.Dense(32, activation='relu')(dense_output)
#     output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_output)
#     model = tf.keras.Model(inputs=[input_data, extra_value, hot_encoded_value], outputs=output)
#
#     # Compile the model
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     # Print the summary of the model
#     model.fit(
#         [x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded], y_train,
#         epochs=300,
#         verbose=1,
#         validation_data=([x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded], y_cv),
#         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)],
#         batch_size=1024,
#         class_weight=class_weight
#
#     )
#
#     yhat_train = model.predict([x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded])
#     predictions_train = np.where(yhat_train >= 0.5, 1, 0)
#     print(classification_report(y_train, predictions_train))
#
#     yhat_cv = model.predict([x_cv_components_scaled_padded, x_cv_sizes_scaled, x_cv_n_patches_encoded])
#     predictions_cv = np.where(yhat_cv >= 0.5, 1, 0)
#     print(classification_report(y_cv, predictions_cv))
#
#     plotPrecisionRecall(yhat_cv, y_cv, "model" + str(i))
#
#     all_predictions.extend(yhat_cv)
#     all_labels.extend(y_cv)
#     yhat_groups.append(yhat_cv.reshape(-1))
#     y_train_groups.append(y_train)
#
# plotPrecisionRecall(all_predictions, all_labels, "all models together")


#

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
for n_layers_component in range(4, 6):
    for n_layers_size in range(2, 3):
        for n_layers_n_patches in range(2, 3):
            for n_layers_final in range(3, 4):
                for n_early_stopping_epochs in range(5, 6):
                    for batch_size in [pow(2, 10 + i) for i in range(1)]:
                        model = utils.buildModel(n_layers_component, n_layers_size, n_layers_n_patches, n_layers_final)
                        # Compile the model
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        architectureAucs = []
                        for i in range(len(dictsForTraining)):
                            print(n_layers_component,n_layers_size,n_layers_n_patches,n_layers_final,n_early_stopping_epochs,batch_size,i)
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
                            class_weights = utils.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
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
                        totalAucs.append(((n_layers_component, n_layers_size, n_layers_n_patches, n_layers_final),
                                         pr_auc))
                        print(((n_layers_component, n_layers_size, n_layers_n_patches, n_layers_final,n_early_stopping_epochs,batch_size),
                                         pr_auc))

utils.saveAsPickle(allArchitecturesAucs,
             os.path.join(path.aggregateFunctionMLPDir, os.path.join('gridSearch', 'allArchitecturesAucs')))
utils.saveAsPickle(totalAucs, os.path.join(path.aggregateFunctionMLPDir, os.path.join('gridSearch', 'totalAucs')))

