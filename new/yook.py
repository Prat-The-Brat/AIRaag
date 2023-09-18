# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# make sure ffmpeg is installed as backend.
# !apt install -y ffmpeg

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from zstandard import TARGETLENGTH_MAX
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))
        pass

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Dataset location
SOURCE_PATH = '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/'

# Path to labels and processed data file, json format.
JSON_PATH = '/kaggle/working/data.json'

# Sampling rate.
sr = 22050

# Let's make sure all files have the same amount of samples and pick a duration right under 30 seconds.
TOTAL_SAMPLES = 29 * sr

# The dataset contains 999 files. Lets make it bigger.
# X amount of slices => X times more training examples.
NUM_SLICES = 10
SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)


def preprocess_data(source_path, json_path):

    # Let's create a dictionary of labels and processed data.
    mydict = {
        "labels": [],
        "mfcc": []
    }

    # Let's browse each file, slice it and generate the 13 band mfcc for each slice.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(source_path)):
        for file in filenames:
            # exclude a corrupted wav file that makes everything crash.
            if os.path.join(dirpath, file) != '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/jazz/jazz.00054.wav':
                song, sr = librosa.load(
                    os.path.join(dirpath, file), duration=29)
                for s in range(NUM_SLICES):
                    start_sample = SAMPLES_PER_SLICE * s
                    end_sample = start_sample + SAMPLES_PER_SLICE
                    mfcc = librosa.feature.mfcc(
                        y=song[start_sample:end_sample], sr=sr, n_mfcc=13)
                    mfcc = mfcc.T
                    mydict["labels"].append(i-1)
                    mydict["mfcc"].append(mfcc.tolist())
            else:
                pass

    # Let's write the dictionary in a json file.
    with open(json_path, 'w') as f:
        json.dump(mydict, f)
    f.close()


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    f.close()

    # Let's load our data into numpy arrays for TensorFlow compatibility.
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def prepare_datasets(inputs, targets, split_size):

    # Creating a validation set and a test set.
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs, targets, test_size=split_size)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs_train, targets_train, test_size=split_size)

    # Our CNN model expects 3D input shape.
    inputs_train = inputs_train[..., np.newaxis]
    inputs_val = inputs_val[..., np.newaxis]
    inputs_test = inputs_test[..., np.newaxis]

    return inputs_train, inputs_val, inputs_test, targets_train, targets_val, targets_test


def design_model(input_shape):

    # Let's design the model architecture.
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(
            len(np.unique(TARGETLENGTH_MAX)), activation='softmax')
    ])

    return model


def make_prediction(model, X, y, idx):

    genre_dict = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock",
    }

    predictions = model.predict(X)
    genre = np.argmax(predictions[idx])

    print("\n---Now testing the model for one audio file---\nThe model predicts: {}, and ground truth is: {}.\n".format(
        genre_dict[genre], genre_dict[y[idx]]))


def plot_performance(hist):

    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    if __name__ == "__main__":

        preprocess_data(source_path=SOURCE_PATH, json_path=JSON_PATH)

        inputs, targets = load_data(json_path=JSON_PATH)

        Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(
            inputs, targets, 0.2)

        input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)
        model = design_model(input_shape)

        # Selection of the optimizer, loss type and metrics for performance evaluation.
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc']
                      )

        model.summary()

        # Training the model.
        history = model.fit(Xtrain, ytrain,
                            validation_data=(Xval, yval),
                            epochs=30,
                            batch_size=32
                            )

        plot_performance(history)

        # Testing the model on never seen before data.
        make_prediction(model, Xtest, ytest, 24)
