# ==============================
# Sentiment Analysis using LSTM (Softmax - 3 Classes)
# ==============================

import pandas as pd
import numpy as np
import os
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# ==============================
# Load Data (NO COLUMN NAMES)
# ==============================

data = pd.read_csv("twitter_training.csv", header=None)
data.columns = ["id", "topic", "sentiment", "text"]
data.drop(columns=["id"], inplace=True)

# Drop missing text
data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str)

# ==============================
# Basic Twitter Text Cleaning
# ==============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)     # remove URLs
    text = re.sub(r"@\w+", "", text)               # remove mentions
    text = re.sub(r"#", "", text)                  # remove hashtag symbol
    text = re.sub(r"[^a-z\s]", "", text)           # remove emojis & symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

# ==============================
# Encode Sentiment into 3 Classes
# Positive | Negative | Rest
# ==============================

data["sentiment_grouped"] = data["sentiment"].apply(
    lambda x: x if x in ["Positive", "Negative"] else "Rest"
)

sentiment_onehot = pd.get_dummies(
    data["sentiment_grouped"],
    prefix="sentiment"
)

data = pd.concat([data, sentiment_onehot], axis=1)
data.drop(columns=["sentiment"], inplace=True)

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
# Tokenization & Padding
# ==============================

VOCAB_SIZE = 15000
MAX_LEN = 300

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

print(X_train.shape)
print(Y_train.shape)

# ==============================
# Handle Class Imbalance
# ==============================

y_labels = np.argmax(Y_train, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_labels),
    y=y_labels
)
class_weight_dict = dict(enumerate(class_weights))

# ==============================
# LSTM Model (BiLSTM + Softmax)
# ==============================

model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
model.add(LSTM(64, dropout=0.3))
model.add(Dense(3, activation="softmax"))

model.summary()

# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Train Model
# ==============================

model.fit(
    X_train,
    Y_train,
    epochs=8,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict
)

# ==============================
# Evaluate Model
# ==============================

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# ==============================
# Confusion Matrix
# ==============================

# Convert one-hot labels to class indices
y_true = np.argmax(Y_test, axis=1)

# Predict class probabilities
y_pred_probs = model.predict(X_test, verbose=0)

# Convert predictions to class indices
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Positive", "Negative", "Rest"],
    yticklabels=["Positive", "Negative", "Rest"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# Classification Report
# ==============================
print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Positive", "Negative", "Rest"]
    )
)


# ==============================
# Prediction Function
# ==============================

label_map = ["Positive", "Negative", "Rest"]

def predict_sentiment(text):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded, verbose=0)
    return label_map[np.argmax(prediction)]

# ==============================
# Example Predictions
# ==============================

examples = [
    "This game is absolutely amazing",
    "Worst experience ever",
    "Just another normal day"
]

for text in examples:
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {predict_sentiment(text)}\n")
