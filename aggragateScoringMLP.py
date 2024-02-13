# for array computations and loading data
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

# custom functions
# import utils

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)


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
    x_train, x_, y_train, y_ = train_test_split(x, y.numpy(), test_size=0.40, random_state=1)

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
        size_scaled = scalerSize.transform(tf.reshape(x[i][:, 0], [-1, 1]))
        ubAverages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 1], [-1, 1]))
        nonUbAverages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 2], [-1, 1]))
        x[i] = np.concatenate((size_scaled, ubAverages_scaled, nonUbAverages_scaled), axis=1)


def scaleXValues3D(x_train, x_cv, x_test):
    # Scale the features using the z-score
    allTupleSizesTrain = np.concatenate([tuples[:, 0] for tuples in x_train if tuples.shape != (0,)])
    allTupleUbAveragesTrain = np.concatenate([tuples[:, 1] for tuples in x_train if tuples.shape != (0,)])
    scalerSize = StandardScaler()
    scalerAverageUbBinding = StandardScaler()
    scalerSize.fit(allTupleSizesTrain.reshape((-1, 1)))
    scalerAverageUbBinding.fit(allTupleUbAveragesTrain.reshape((-1, 1)))
    Scale3DUtil(x_train, scalerSize, scalerAverageUbBinding)
    Scale3DUtil(x_cv, scalerSize, scalerAverageUbBinding)
    Scale3DUtil(x_test, scalerSize, scalerAverageUbBinding)
    return x_train, x_cv, x_test


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
# maxNumberOfPatches = 10
# x_train, x_cv, x_test, y_train, y_cv, y_test = divideTrainValidationTest(x, y)
# x_train_scaled, x_cv_scaled, x_test_scaled = scaleXValues3D(x_train, x_cv, x_test)
# x_train, x_cv, x_test = padXValues(x_train, x_cv, x_test, maxNumberOfPatches)
#
# print(f"the shape of the training set (input) is: {x_train.shape}")
# print(f"the shape of the training set (target) is: {y_train.shape}\n")
# print(f"the shape of the validation set (input) is: {x_cv.shape}")
# print(f"the shape of the validation set (target) is: {y_cv.shape}\n")
# print(f"the shape of the test set (input) is: {x_test.shape}")
# print(f"the shape of the test set (target) is: {y_test.shape}")
#
# # Initialize lists that will contain the errors for each model
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)
model_2.fit(
    x_scaled_padded_train, labels_train,
    epochs=5,
    verbose=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)],
    batch_size=128

)
yhat_train = model_2.predict(x_scaled_padded_train)
predictions_train = np.where(yhat_train >= 0.5, 1, 0)
print(classification_report(labels_train, predictions_train))

yhat = model_2.predict(x_scaled_padded_test)
predictions_test = np.where(yhat >= 0.5, 1, 0)
print(classification_report(labels_test, predictions_test))
