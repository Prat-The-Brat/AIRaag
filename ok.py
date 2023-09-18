import os
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import soundfile as sf

# Dataset location
SOURCE_PATH = 'Data/genres_original/'

# Path to labels and processed data file, json format.
JSON_PATH = 'data.json'

# Sampling rate.
sr = 22050

# Let's make sure all files have the same amount of samples, pick a duration right under 30 seconds.
TOTAL_SAMPLES = 29 * sr

# The dataset contains 999 files. Let's make it bigger.
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
            try:
                song, sr = librosa.load(
                    os.path.join(dirpath, file), duration=29)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue

            for s in range(NUM_SLICES):
                start_sample = SAMPLES_PER_SLICE * s
                end_sample = start_sample + SAMPLES_PER_SLICE
                mfcc = librosa.feature.mfcc(
                    y=song[start_sample:end_sample], sr=sr, n_mfcc=13)
                mfcc = mfcc.T
                mydict["labels"].append(i-1)
                mydict["mfcc"].append(mfcc.tolist())

    # Let's write the dictionary in a json file.
    with open(json_path, 'w') as f:
        json.dump(mydict, f)


def load_data(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

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
        tf.keras.layers.Dense(len(np.unique(targets)), activation='softmax')
    ])

    return model


def make_prediction(model, X, y, idx):

    unique_genres = np.unique(y)
    genre_dict = {i: genre for i, genre in enumerate(unique_genres)}

    predictions = model.predict(X)
    genre = np.argmax(predictions[idx])

    print("\n---Now testing the model for one audio file---\nThe model predicts: {}, and ground truth is: {}.\n".format(
        genre_dict[genre], genre_dict[y[idx]]))


def plot_performance(hist):

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
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


# ... (Previous code remains unchanged)

if __name__ == "__main__":

    preprocess_data(source_path=SOURCE_PATH, json_path=JSON_PATH)

    inputs, targets = load_data(json_path=JSON_PATH)

    Xtrain, Xval, Xtest, ytrain, yval, ytest = prepare_datasets(
        inputs, targets, 0.2)

    input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1)

    # List of parameter combinations to try
    parameters = [
        {'optimizer': tf.keras.optimizers.legacy.RMSprop(lr=0.001)},
        {'optimizer': tf.keras.optimizers.legacy.Adam(learning_rate=0.001)},
        # Add more parameter combinations as needed
    ]

    for params in parameters:
        print(f"\nTraining model with parameters: {params}")

        # Create a new model with the current parameter set
        model = design_model(input_shape)

        # Update model's optimizer with the current parameter value
        model.compile(optimizer=params['optimizer'],
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(Xtrain, ytrain,
                            validation_data=(Xval, yval),
                            epochs=30,
                            batch_size=32,
                            verbose=0  # Set to 1 for progress updates during training
                            )

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(Xtest, ytest, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Generate the confusion matrix
        y_pred = model.predict(Xtest)
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(ytest, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks(np.arange(len(np.unique(targets))), np.unique(targets))
        plt.yticks(np.arange(len(np.unique(targets))), np.unique(targets))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Print the classification report
        class_report = classification_report(
            ytest, y_pred, target_names=np.unique(targets).astype(str))
        print("Classification Report:")
        print(class_report)

        # Optionally, save the model with the current parameters
        # model.save(f'model_with_params_{params}.h5')
