"""
Music Genre Classification (Local + Google Drive Download)
----------------------------------------------------------
Pipeline:
1) Download GTZAN dataset from Google Drive (gdown)
2) Extract dataset
3) MFCC feature extraction
4) ANN, Regularized ANN, CNN models
"""

# ============================
# Imports
# ============================
import os
import math
import json
import zipfile
import numpy as np
import librosa
import matplotlib.pyplot as plt
import gdown

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout,
    Conv2D, MaxPool2D, BatchNormalization
)
import tensorflow.keras as keras

# ============================
# Google Drive Download Config
# ============================
GDRIVE_FILE_ID = "1f1MecRibxjyVQ8dGWhH0kffVDGVk3Dpn"   # 🔴 CHANGE THIS
ZIP_PATH = "gtzan.zip"
DATASET_ROOT = "datasets"
DATASET_PATH = "datasets/genres_original"
JSON_PATH = "datasets/data.json"


# ============================
# Audio Parameters
# ============================
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# ============================
# Download & Extract Dataset
# ============================
def download_and_extract_gtzan():
    if not os.path.exists(DATASET_PATH):
        print("Downloading GTZAN dataset from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_ROOT)

        print("GTZAN dataset ready.")
    else:
        print("GTZAN dataset already exists. Skipping download.")

# ============================
# MFCC Feature Extraction
# ============================
def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048,
              hop_length=512, num_segments=10):

    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_vectors = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath != dataset_path:
            label = os.path.basename(dirpath)
            data["mapping"].append(label)
            print(f"Processing genre: {label}")

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"Skipping corrupted file: {file_path}")
                    continue

                for s in range(num_segments):
                    start = samples_per_segment * s
                    finish = start + samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        y=signal[start:finish],
                        sr=sr,
                        n_mfcc=n_mfcc,
                        n_fft=n_fft,
                        hop_length=hop_length
                    ).T

                    if len(mfcc) == expected_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

# ============================
# Load Dataset
# ============================
def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["mfcc"]), np.array(data["labels"])

# ============================
# Plot Training History
# ============================
def plot_history(history):
    fig, axs = plt.subplots(2, figsize=(12, 8))

    axs[0].plot(history.history["accuracy"], label="Train")
    axs[0].plot(history.history["val_accuracy"], label="Validation")
    axs[0].set_title("Accuracy")
    axs[0].legend()

    axs[1].plot(history.history["loss"], label="Train")
    axs[1].plot(history.history["val_loss"], label="Validation")
    axs[1].set_title("Loss")
    axs[1].legend()

    plt.show()

# ============================
# ANN Model
# ============================
def train_ann(X_train, X_test, y_train, y_test):
    model = Sequential([
        Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )

    plot_history(history)

# ============================
# Regularized ANN
# ============================
def train_regularized_ann(X_train, X_test, y_train, y_test):
    model = Sequential([
        Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(512, activation="relu",
              kernel_regularizer=keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(256, activation="relu",
              kernel_regularizer=keras.regularizers.l2(0.003)),
        Dropout(0.3),
        Dense(64, activation="relu",
              kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )

    plot_history(history)

# ============================
# CNN Dataset Prep
# ============================
def prepare_cnn_data():
    X, y = load_data(JSON_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    return X_train, X_val, y_train, y_val

# ============================
# CNN Model
# ============================
def train_cnn(X_train, X_val, y_train, y_val):
    model = Sequential([
        Conv2D(64, (3, 3), activation="relu", input_shape=X_train.shape[1:]),
        MaxPool2D((3, 3), strides=(2, 2), padding="same"),
        BatchNormalization(),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D((3, 3), strides=(2, 2), padding="same"),
        BatchNormalization(),

        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32
    )

    plot_history(history)

# ============================
# Main
# ============================
if __name__ == "__main__":

    download_and_extract_gtzan()

    if not os.path.exists(JSON_PATH):
        print("Extracting MFCC features...")
        save_mfcc(DATASET_PATH, JSON_PATH)

    X, y = load_data(JSON_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("Training ANN...")
    train_ann(X_train, X_test, y_train, y_test)

    print("Training Regularized ANN...")
    train_regularized_ann(X_train, X_test, y_train, y_test)

    print("Training CNN...")
    X_train, X_val, y_train, y_val = prepare_cnn_data()
    train_cnn(X_train, X_val, y_train, y_val)
