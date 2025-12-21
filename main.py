# ==============================
# Sentiment Analysis using LSTM (Softmax - 3 Classes)
# ==============================

import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
TF_ENABLE_ONEDNN_OPTS=0
# ==============================
# Load Data (NO COLUMN NAMES)
# ==============================

# Read CSV without headers
data = pd.read_csv("twitter_training.csv", header=None)

# Manually assign column names
data.columns = ["id", "topic", "sentiment", "text"]

data.drop(columns=['id'], inplace=True)

# Drop rows where text is missing
data = data.dropna(subset=["text"])

# Ensure text column is string
data["text"] = data["text"].astype(str)

# print(data.sample(5))
# print(data.shape)
# print(data["sentiment"].value_counts())

# ==============================
# Encode Sentiment into 3 Classes
# Positive | Negative | Rest
# ==============================

data["sentiment_grouped"] = data["sentiment"].apply(
    lambda x: x if x in ["Positive", "Negative"] else "Rest"
)

# One-hot encoding
sentiment_onehot = pd.get_dummies(
    data["sentiment_grouped"],
    prefix="sentiment"
)

data = pd.concat([data, sentiment_onehot], axis=1)

# print(sentiment_onehot.head())
data.drop(columns=['sentiment'], inplace=True)
# print(data.sample(5))


# ==============================
# Train-Test Split
# ==============================

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# print(train_data.shape)
# print(test_data.shape)

# ==============================
# Tokenization & Padding
# ==============================

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["text"])

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train_data["text"]),
    maxlen=200
)

X_test = pad_sequences(
    tokenizer.texts_to_sequences(test_data["text"]),
    maxlen=200
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
# LSTM Model (Softmax)
# ==============================

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
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
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# ==============================
# Evaluate Model
# ==============================

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# ==============================
# Prediction Function
# ==============================

label_map = ["Positive", "Negative", "Rest"]

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
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
