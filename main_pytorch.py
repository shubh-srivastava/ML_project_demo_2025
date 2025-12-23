# ==============================
# Sentiment Analysis using LSTM (PyTorch)
# CUDA-Optimized for Native Windows + NVIDIA GPU
# ==============================

import pandas as pd
import numpy as np
import re
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# CUDA GPU Configuration
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx) for faster training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

# ==============================
# Load Data
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
# ==============================

label_map = {"Positive": 0, "Negative": 1, "Rest": 2}
label_names = ["Positive", "Negative", "Rest"]

data["sentiment_grouped"] = data["sentiment"].apply(
    lambda x: x if x in ["Positive", "Negative"] else "Rest"
)
data["label"] = data["sentiment_grouped"].map(label_map)

# ==============================
# Train-Test Split
# ==============================

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# ==============================
# Tokenization
# ==============================

VOCAB_SIZE = 15000
MAX_LEN = 300  # Same as TensorFlow

# Build vocabulary from training data
word_counts = Counter()
for text in train_data["text"]:
    word_counts.update(text.split())

# Create word to index mapping (reserve 0 for padding, 1 for unknown)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(VOCAB_SIZE - 2))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode_text(text):
    tokens = text.split()
    encoded = [vocab.get(word, 1) for word in tokens]  # 1 = <UNK>
    # Pad or truncate (pre-padding like Keras default)
    if len(encoded) < MAX_LEN:
        # Pre-padding: add zeros at the BEGINNING (same as Keras pad_sequences default)
        encoded = [0] * (MAX_LEN - len(encoded)) + encoded
    else:
        # Truncate from the beginning (same as Keras default truncating='pre')
        encoded = encoded[-MAX_LEN:]
    return encoded

# ==============================
# PyTorch Dataset
# ==============================

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [encode_text(text) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

train_dataset = SentimentDataset(train_data["text"].tolist(), train_data["label"].tolist())
test_dataset = SentimentDataset(test_data["text"].tolist(), test_data["label"].tolist())

# Same batch size as TensorFlow
BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ==============================
# Handle Class Imbalance
# ==============================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_data["label"].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ==============================
# BiLSTM Model
# ==============================

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # BiLSTM layer (128 units) - matches TensorFlow's Bidirectional(LSTM(128, dropout=0.3))
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        # Second LSTM (64 units) - matches TensorFlow's LSTM(64, dropout=0.3)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim // 2, batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(dropout)
        # Dense layer - matches TensorFlow's Dense(3)
        self.fc = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout_rate = dropout

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # BiLSTM with return_sequences=True equivalent
        # Apply dropout to input (similar to Keras LSTM dropout parameter)
        if self.training:
            embedded = nn.functional.dropout(embedded, p=self.dropout_rate)
        lstm1_out, _ = self.lstm1(embedded)  # (batch, seq_len, hidden_dim * 2)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM - apply dropout to input
        if self.training:
            lstm1_out_dropped = nn.functional.dropout(lstm1_out, p=self.dropout_rate)
        else:
            lstm1_out_dropped = lstm1_out
        lstm2_out, _ = self.lstm2(lstm1_out_dropped)  # (batch, seq_len, hidden_dim // 2)

        # Get the last timestep output (return_sequences=False equivalent)
        last_output = lstm2_out[:, -1, :]  # (batch, hidden_dim // 2)

        out = self.fc(last_output)
        return out

model = BiLSTMClassifier(
    vocab_size=VOCAB_SIZE,
    embed_dim=128,
    hidden_dim=128,
    num_classes=3,
    dropout=0.3
).to(device)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==============================
# Training Setup
# ==============================

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Same as TensorFlow default

# Mixed precision for faster training on RTX GPUs
scaler = torch.amp.GradScaler('cuda')

# Print class distribution and weights
print(f"\nClass distribution in training data:")
print(train_data["sentiment_grouped"].value_counts())
print(f"\nClass weights: {class_weights}")

# ==============================
# Training Loop
# ==============================

EPOCHS = 20
best_val_loss = float('inf')
patience = 2  # Same as TensorFlow EarlyStopping
patience_counter = 0

# Split train into train/val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast('cuda'):
            outputs = model(texts)
            loss = criterion(outputs, labels)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(texts)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pt"))

# ==============================
# Evaluate Model
# ==============================

model.eval()
test_loss = 0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)

        outputs = model(texts)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
test_acc = test_correct / test_total

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ==============================
# Classification Report
# ==============================

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=label_names))

# ==============================
# Learning Curves
# ==============================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig("learning_curves.png")
plt.show()

# ==============================
# Prediction Function
# ==============================

def predict_sentiment(text):
    model.eval()
    text = clean_text(text)
    encoded = torch.tensor([encode_text(text)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(encoded)
        pred = output.argmax(1).item()
    return label_names[pred]

# ==============================
# Example Predictions
# ==============================

examples = [
    "This game is absolutely amazing",
    "Worst experience ever",
    "Just another normal day"
]

print("\nExample Predictions:")
for text in examples:
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {predict_sentiment(text)}\n")

# ==============================
# Save Model & Tokenizer
# ==============================

torch.save(model.state_dict(), "sentiment_bilstm_model.pt")

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_names, f)

print("Model, vocabulary, and label map saved successfully!")
