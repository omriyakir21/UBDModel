# for array computations and loading data
import os

from PIL.ImageOps import pad
from sklearn.utils import compute_class_weight

import path
import numpy as np

# for building linear regression models and preparing data
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import pickle
# for building and training neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Reshape, Masking
from keras import backend

# custom functions
# import utils

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
maxNumberOfPatches = 10


class GlobalSumPooling(GlobalAveragePooling1D):
    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        if mask is not None:
            mask = tf.cast(mask, inputs[0].dtype)
            mask = tf.expand_dims(
                mask, 2 if self.data_format == "channels_last" else 1
            )
            inputs *= mask
            return backend.sum(
                inputs, axis=steps_axis, keepdims=self.keepdims
            )


def createRandomDataSet(size):
    # Set the number of matrices (m) in the list
    m_matrices = size  # Adjust as needed

    # Create a list to store the matrices
    matrices_list = []

    # Generate random matrices for each index in the list
    for _ in range(m_matrices):
        k_i = np.random.randint(5, 21)  # Randomly sample k_i between 5 and 20
        matrix_i = np.random.rand(k_i, 3)  # Create a random matrix of dimensions (k_i, 3)
        matrices_list.append(tf.constant(matrix_i))

    # Generate random binary labels (0 or 1) for each vector
    random_labels_np = np.random.randint(2, size=size)

    # Convert NumPy arrays to TensorFlow tensors
    random_labels_tf = tf.reshape(tf.constant(random_labels_np), (-1, 1))
    return matrices_list, random_labels_tf


def sortPatches(x):
    try:

        for i in range(len(x)):
            # Get indices that would sort the tensor along the second column
            if x[i].shape != (0,):
                sorted_indices = tf.argsort(x[i][:, 1])
                # Use tf.gather to rearrange rows based on the sorted indices
                sorted_tensor = tf.gather(x[i], sorted_indices)
                x[i] = sorted_tensor
    except Exception:
        print(1)


def divideTrainValidationTest(x, y):
    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
    # Delete temporary variables
    del x_, y_
    return x_train, x_cv, x_test, y_train, y_cv, y_test


def padXValues(x_train, x_cv, x_test, maxNumberOfPatches):
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, padding="post", maxlen=maxNumberOfPatches, dtype='float32'
    )
    x_cv = tf.keras.preprocessing.sequence.pad_sequences(
        x_cv, padding="post", maxlen=maxNumberOfPatches, dtype='float32'
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, padding="post", maxlen=maxNumberOfPatches, dtype='float32'
    )
    return x_train, x_cv, x_test


def Scale2DUtil(x, scalerSize, scalerAverageUbBinding):
    for i in range(len(x)):
        if x[i].shape != (0,):
            size_scaled = scalerSize.transform(tf.reshape(x[i][:, 0], [-1, 1]))
            averages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 1], [-1, 1]))
            x[i] = np.concatenate((size_scaled, averages_scaled), axis=1)


def scaleXValues2D(x_train, x_cv, x_test):
    # Scale the features using the z-score
    allTupleSizesTrain = np.concatenate([tuples[:, 0] for tuples in x_train if tuples.shape != (0,)])
    allTupleUbAveragesTrain = np.concatenate([tuples[:, 1] for tuples in x_train if tuples.shape != (0,)])
    scalerSize = StandardScaler()
    scalerAverageUbBinding = StandardScaler()
    scalerSize.fit(allTupleSizesTrain.reshape((-1, 1)))
    scalerAverageUbBinding.fit(allTupleUbAveragesTrain.reshape((-1, 1)))
    Scale2DUtil(x_train, scalerSize, scalerAverageUbBinding)
    Scale2DUtil(x_cv, scalerSize, scalerAverageUbBinding)
    Scale2DUtil(x_test, scalerSize, scalerAverageUbBinding)
    return x_train, x_cv, x_test


def Scale3DUtil(x, scalerSize, scalerAverageUbBinding):
    for i in range(len(x)):
        if x[i].shape == (0,):
            continue
        size_scaled = scalerSize.transform(tf.reshape(x[i][:, 0], [-1, 1]))
        ubAverages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 1], [-1, 1]))
        nonUbAverages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 2], [-1, 1]))
        x[i] = np.concatenate((size_scaled, ubAverages_scaled, nonUbAverages_scaled), axis=1)


def scaleXComponents3D(x_train_components, x_cv_components, x_test_components):
    # Scale the features using the z-score
    allTupleSizesTrain = np.concatenate([tuples[:, 0] for tuples in x_train_components if tuples.shape != (0,)])
    allTupleUbAveragesTrain = np.concatenate([tuples[:, 1] for tuples in x_train_components if tuples.shape != (0,)])
    scalerSize = StandardScaler()
    scalerAverageUbBinding = StandardScaler()
    scalerSize.fit(allTupleSizesTrain.reshape((-1, 1)))
    scalerAverageUbBinding.fit(allTupleUbAveragesTrain.reshape((-1, 1)))
    Scale3DUtil(x_train_components, scalerSize, scalerAverageUbBinding)
    Scale3DUtil(x_cv_components, scalerSize, scalerAverageUbBinding)
    Scale3DUtil(x_test_components, scalerSize, scalerAverageUbBinding)
    return x_train_components, x_cv_components, x_test_components


def getScaleXSizes3D(x_train_sizes, x_cv_sizes, x_test_sizes):
    # Scale the features using the z-score
    scalerSize = StandardScaler()
    x_train_sizes_scaled = scalerSize.fit_transform(x_train_sizes.reshape((-1, 1)))
    x_cv_sizes_scaled = scalerSize.transform(x_cv_sizes.reshape((-1, 1)))
    x_test_sizes_scaled = scalerSize.transform(x_test_sizes.reshape((-1, 1)))
    return x_train_sizes_scaled, x_cv_sizes_scaled, x_test_sizes_scaled


def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object


def build_models():
    tf.random.set_seed(20)

    model_1 = Sequential(
        [tf.keras.Input(shape=(maxNumberOfPatches, 3)),
         Masking(mask_value=0.0),
         Dense(16, activation='relu'),
         GlobalAveragePooling1D(data_format='channels_last'),
         Dense(1, activation='linear')
         ],
        name='model_1'
    )

    model_2 = Sequential(
        [
            tf.keras.Input(shape=(maxNumberOfPatches, 3)),
            Masking(mask_value=0.0),
            Dense(25, activation='relu'),
            Dense(16, activation='relu'),
            GlobalAveragePooling1D(data_format='channels_last'),
            Dense(1, activation='linear')
        ],
        name='model_2'
    )

    model_3 = Sequential(
        [
            tf.keras.Input(shape=(maxNumberOfPatches, 3)),
            Masking(mask_value=0.0),
            Dense(32, activation='relu'),
            Dense(25, activation='relu'),
            Dense(16, activation='relu'),
            GlobalAveragePooling1D(data_format='channels_last'),
            Dense(1, activation='linear')
        ],
        name='model_3'
    )

    model_list = [model_1, model_2, model_3]

    return model_list


# # TODO import x and y
# x, y = createRandomDataSet(100)
# sortPatches(x)
def divideXData(x):
    components = [np.array(tup[2]) for tup in x]
    sizes = np.array([tup[3] for tup in x])
    n_patches = np.array([tup[4] for tup in x])

    return components, sizes, n_patches


def hotOneEncodeNPatches(n_patches_array):
    encoded = np.zeros((n_patches_array.shape[0], maxNumberOfPatches + 1))
    encoded[np.arange(n_patches_array.size), np.minimum(n_patches_array,
                                                        np.full(n_patches_array.shape, maxNumberOfPatches))] = 1
    return encoded


def createDataForTraining():
    labels = loadPickle(os.path.join(path.mainProjectDir, os.path.join('aggregateFunctionMLP', 'labels3d.pkl')))
    tuplesLen3 = loadPickle(
        os.path.join(path.mainProjectDir, os.path.join('aggregateFunctionMLP', 'allTuplesListsOfLen3.pkl')))
    x_train, x_cv, x_test, y_train, y_cv, y_test = divideTrainValidationTest(tuplesLen3, labels)
    x_train_components, x_train_sizes, x_train_n_patches = divideXData(x_train)
    x_cv_components, x_cv_sizes, x_cv_n_patches = divideXData(x_cv)
    x_test_components, x_test_sizes, x_test_n_patches = divideXData(x_test)
    x_train_n_patches_encoded = hotOneEncodeNPatches(x_train_n_patches)
    x_cv_n_patches_encoded = hotOneEncodeNPatches(x_cv_n_patches)
    x_test_n_patches_encoded = hotOneEncodeNPatches(x_test_n_patches)
    sortPatches(x_train_components)
    sortPatches(x_cv_components)
    sortPatches(x_test_components)
    scaleXComponents3D(x_train_components, x_cv_components, x_test_components)
    x_train_sizes_scaled, x_cv_sizes_scaled, x_test_sizes_scaled = getScaleXSizes3D(x_train_sizes, x_cv_sizes,
                                                                                    x_test_sizes)
    x_train_components_scaled_padded, x_cv_components_scaled_padded, x_test_components_scaled_padded = padXValues(
        x_train_components, x_cv_components, x_test_components, maxNumberOfPatches)
    allInfoDict = {'x_train': x_train, 'x_cv': x_cv, 'x_test': x_test, 'y_train': y_train, 'y_cv': y_cv,
                   'y_test': y_test}
    dictForTraining = {'x_train_components_scaled_padded': x_train_components_scaled_padded,
                       'x_cv_components_scaled_padded': x_cv_components_scaled_padded,
                       'x_test_components_scaled_padded': x_test_components_scaled_padded,
                       'x_train_sizes_scaled': x_train_sizes_scaled, 'x_cv_sizes_scaled': x_cv_sizes_scaled,
                       'x_test_sizes_scaled': x_test_sizes_scaled,
                       'x_train_n_patches_encoded': x_train_n_patches_encoded,
                       'x_cv_n_patches_encoded': x_cv_n_patches_encoded,
                       'x_test_n_patches_encoded': x_test_n_patches_encoded,
                       'y_train': y_train, 'y_cv': y_cv,
                       'y_test': y_test
                       }

    saveAsPickle(allInfoDict,
                 os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining1902', 'allInfoDict')))
    saveAsPickle(dictForTraining,
                 os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining1902', 'dictForTraining')))


# createDataForTraining()
dictForTraining = loadPickle(
    os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining1902', 'dictForTraining.pkl')))
x_train_components_scaled_padded = dictForTraining['x_train_components_scaled_padded']
x_cv_components_scaled_padded = dictForTraining['x_cv_components_scaled_padded']
x_test_components_scaled_padded = dictForTraining['x_test_components_scaled_padded']
x_train_sizes_scaled = dictForTraining['x_train_sizes_scaled']
x_cv_sizes_scaled = dictForTraining['x_cv_sizes_scaled']
x_test_sizes_scaled = dictForTraining['x_test_sizes_scaled']
x_train_n_patches_encoded = dictForTraining['x_train_n_patches_encoded']
x_cv_n_patches_encoded = dictForTraining['x_cv_n_patches_encoded']
x_test_n_patches_encoded = dictForTraining['x_test_n_patches_encoded']
y_train = dictForTraining['y_train']
y_cv = dictForTraining['y_cv']
y_test = dictForTraining['y_test']
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# Convert class weights to a dictionary
class_weight = {i: class_weights[i] for i in range(len(class_weights))}

model = Sequential(
    [
        tf.keras.Input(shape=(maxNumberOfPatches, 3)),
        Masking(mask_value=0.0),
        Dense(36,activation='relu'),
        Dense(25, activation='relu'),
        Dense(16, activation='relu'),
        GlobalSumPooling(data_format='channels_last'),
        Dense(1, activation='sigmoid')
    ],
    name='model_2'
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# Convert class weights to a dictionary
class_weight = {i: class_weights[i] for i in range(len(class_weights))}

model.fit(
    x_train_components_scaled_padded, y_train,
    epochs=300,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)],
    batch_size=1024,
    class_weight=class_weight

)

yhat_train = model.predict(x_train_components_scaled_padded)
predictions_train = np.where(yhat_train >= 0.5, 1, 0)
print(classification_report(y_train, predictions_train))


# Define the input shape
input_shape = (10, 3)

# Define the input layers
input_data = tf.keras.Input(shape=input_shape, name='sensor_input')
extra_value = tf.keras.Input(shape=(1,), name='extra_value_input')
hot_encoded_value = tf.keras.Input(shape=(11,), name='hot_encoded_value_input')

# Masking layer to handle sequences with a padding value of 0.0
masked_input = tf.keras.layers.Masking(mask_value=0.0)(input_data)

# Pass through dense layers with ReLU activation
dense_output = tf.keras.layers.Dense(64, activation='relu')(masked_input)
dense_output = tf.keras.layers.Dense(32, activation='relu')(dense_output)

# Global Average Pooling
global_pooling_output = GlobalSumPooling(data_format='channels_last')(dense_output)

# Concatenate the global pooling output with extra value and hot encoded value
concatenated_output = tf.keras.layers.Concatenate()([global_pooling_output, extra_value, hot_encoded_value])

# Pass through dense layers with ReLU activation
dense_output = tf.keras.layers.Dense(64, activation='relu')(concatenated_output)
dense_output = tf.keras.layers.Dense(32, activation='relu')(dense_output)

# Output layer with sigmoid activation for binary classification
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_output)

# Define the model
model = tf.keras.Model(inputs=[input_data, extra_value, hot_encoded_value], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train_sizes_scaled = dictForTraining['x_train_sizes_scaled']
x_cv_sizes_scaled = dictForTraining['x_cv_sizes_scaled']
x_test_sizes_scaled = dictForTraining['x_test_sizes_scaled']
x_train_n_patches_encoded = dictForTraining['x_train_n_patches_encoded']

# Print the summary of the model
model.summary()
model.fit(
    [x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded], y_train,
    epochs=300,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)],
    batch_size=1024,
    class_weight=class_weight

)

yhat_train = model.predict([x_train_components_scaled_padded, x_train_sizes_scaled, x_train_n_patches_encoded])
predictions_train = np.where(yhat_train >= 0.5, 1, 0)
print(classification_report(y_train, predictions_train))


# nn_train_accuracies = []
# nn_cv_accuracies = []


# Build the models
# nn_models = build_models()


# # Loop over the the models
# for model in nn_models:
#     # Setup the loss and optimizer
#     model.compile(
#         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
#     )
#
#     print(f"Training {model.name}...")
#
#     # Train the model
#     model.fit(
#         x_train_scaled, y_train,
#         epochs=300,
#         verbose=0
#     )
#
#     print("Done!\n")

# Record the Accuracy
# logits = model.predict(x_train_scaled)
# yhat = tf.nn.sigmoid(logits).numpy()
# predictions = np.where(yhat >= 0.5, 1, 0)
# accuracyBool = predictions != y_train
# accuracy_train = np.mean(accuracyBool)
# nn_train_accuracies.append(accuracy_train)
#
# logits = model.predict(x_cv_scaled)
# yhat = tf.nn.sigmoid(logits).numpy()
# predictions = np.where(yhat >= 0.5, 1, 0)
# accuracyBool = predictions != y_cv
# accuracy_cv = np.mean(accuracyBool)
# nn_cv_accuracies.append(accuracy_cv)

# # print results
# print("RESULTS:")
# for model_num in range(len(nn_train_accuracies)):
#     print(
#         f"Model {model_num + 1}: Training accuracy: {nn_train_accuracies[model_num]:.2f}, " +
#         f"CV MSE: {nn_cv_accuracies[model_num]:.2f}"
#     )

def experiment2D():
    labels = loadPickle(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\labels.pkl')
    tuplesLen2 = loadPickle(
        r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\allTuplesListsOfLen2.pkl')
    # labels = labels[:100]
    # tuplesLen2 = tuplesLen2[:100]
    tuplesLen2 = [np.array(tuple) for tuple in tuplesLen2]
    sortPatches(tuplesLen2)
    x_train, x_test, labels_train, labels_test = train_test_split(tuplesLen2, labels, test_size=0.1, random_state=1)
    maxNumberOfPatches = 10
    x_train_scaled, x_cv_scaled, x_test_scaled = scaleXValues2D(x_train, [], x_test)
    x_scaled_padded_train, _, x_scaled_padded_test = padXValues(x_train_scaled, [], x_test_scaled,
                                                                maxNumberOfPatches)
    model_2 = Sequential(
        [
            tf.keras.Input(shape=(maxNumberOfPatches, 2)),
            Masking(mask_value=0.0),
            Dense(25, activation='relu'),
            Dense(16, activation='relu'),
            GlobalAveragePooling1D(data_format='channels_last'),
            Dense(1, activation='sigmoid')
        ],
        name='model_2'
    )
    model_2.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    model_2.fit(
        x_scaled_padded_train, labels_train,
        epochs=300,
        verbose=0
    )

    yhat_train = model_2.predict(x_scaled_padded_train)
    predictions_train = np.where(yhat_train >= 0.5, 1, 0)
    print(classification_report(labels_train, predictions_train))

    yhat = model_2.predict(x_scaled_padded_test)
    predictions_test = np.where(yhat >= 0.5, 1, 0)
    print(classification_report(labels_test, predictions_test))

# experiment2D()


# tuplesLen2 = loadPickle(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\allTuplesListsOfLen2.pkl')
# # labels = labels[:100]
# # tuplesLen2 = tuplesLen2[:100]
# tuplesLen2 = [np.array(tuple) for tuple in tuplesLen2]
# sortPatches(tuplesLen2)
# x_train, x_test, labels_train, labels_test = train_test_split(tuplesLen2, labels, test_size=0.1, random_state=1)
#
# x_train_scaled, x_cv_scaled, x_test_scaled = scaleXValues2D(x_train, [], x_test)
# x_scaled_padded_train, _, x_scaled_padded_test = padXValues(x_train_scaled, [], x_test_scaled,
#                                                             maxNumberOfPatches)


# aggregateFunctionMLPDir = os.path.join(path.mainProjectDir, 'aggregateFunctionMLP')
# x_scaled_padded_train_2d = loadPickle(os.path.join(aggregateFunctionMLPDir, 'x_scaled_padded_train_2d.pkl'))
# x_scaled_padded_test_2d = loadPickle(os.path.join(aggregateFunctionMLPDir, 'x_scaled_padded_test_2d.pkl'))
# labels_train = loadPickle(os.path.join(aggregateFunctionMLPDir, 'labels_train.pkl'))
# labels_test = loadPickle(os.path.join(aggregateFunctionMLPDir, 'labels_test.pkl'))
#
# model_2 = Sequential(
#     [
#         tf.keras.Input(shape=(maxNumberOfPatches, 2)),
#         Masking(mask_value=0.0),
#         Dense(25, activation='relu'),
#         Dense(16, activation='relu'),
#         GlobalSumPooling(data_format='channels_last'),
#         Dense(1, activation='sigmoid')
#     ],
#     name='model_2'
# )
# model_2.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#     metrics=['accuracy']
# )
#
# class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
# # Convert class weights to a dictionary
# class_weight = {i: class_weights[i] for i in range(len(class_weights))}
# model_2.fit(
#     x_scaled_padded_train_2d, labels_train,
#     epochs=300,
#     verbose=1,
#     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)],
#     batch_size=128,
#     class_weight=class_weight
#
# )
# yhat_train = model_2.predict(x_scaled_padded_train_2d)
# predictions_train = np.where(yhat_train >= 0.5, 1, 0)
# print(classification_report(labels_train, predictions_train))
#
# yhat = model_2.predict(x_scaled_padded_test_2d)
# predictions_test = np.where(yhat >= 0.5, 1, 0)
# print(classification_report(labels_test, predictions_test))
