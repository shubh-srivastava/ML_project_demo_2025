# ==============================
# Sentiment Analysis using LSTM (Optimized - 3 Classes)
# ==============================

import pandas as pd
import numpy as np
import os
import re
import pickle

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# Load Data
# ==============================

data = pd.read_csv("twitter_training.csv", header=None)
data.columns = ["id", "topic", "sentiment", "text"]
data.drop(columns=["id"], inplace=True)

data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str)

# ==============================
# Text Cleaning
# ==============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

data["text"] = data["text"].apply(clean_text)

# ==============================
# Encode Sentiment
# ==============================

data["sentiment_grouped"] = data["sentiment"].apply(
    lambda x: x if x in ["Positive", "Negative"] else "Rest"
)

data = pd.concat(
    [data, pd.get_dummies(data["sentiment_grouped"], prefix="sentiment")],
    axis=1
)

# ==============================
# Train-Test Split
# ==============================

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment_grouped"]
)

# ==============================
# Tokenization & Padding (OPTIMIZED)
# ==============================

VOCAB_SIZE = 10000      # ↓ from 15000
MAX_LEN = 120           # ↓ from 300
EMBED_DIM = 100         # ↓ from 128

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_data["text"])

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train_data["text"]),
    maxlen=MAX_LEN
)

X_test = pad_sequences(
    tokenizer.texts_to_sequences(test_data["text"]),
    maxlen=MAX_LEN
)

Y_train = train_data[
    ["sentiment_Positive", "sentiment_Negative", "sentiment_Rest"]
].values

Y_test = test_data[
    ["sentiment_Positive", "sentiment_Negative", "sentiment_Rest"]
].values

# ==============================
# Handle Class Imbalance
# ==============================

y_labels = np.argmax(Y_train, axis=1)
class_weight_dict = dict(enumerate(
    compute_class_weight("balanced", classes=np.unique(y_labels), y=y_labels)
))

# ==============================
# FAST LSTM MODEL (Best Trade-off)
# ==============================

model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
model.add(LSTM(96, dropout=0.3))   # single LSTM (FAST)
model.add(Dense(3, activation="softmax"))

model.summary()

# ==============================
# Compile
# ==============================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Train with Early Stopping
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    Y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)

# ==============================
# Evaluate
# ==============================

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# ==============================
# Confusion Matrix
# ==============================

y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(classification_report(
    y_true, y_pred,
    target_names=["Positive", "Negative", "Rest"]
))

# ==============================
# Save Model (Deployment)
# ==============================

model.save("sentiment_lstm_fast.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_map.pkl", "wb") as f:
    pickle.dump(["Positive", "Negative", "Rest"], f)
