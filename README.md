************************************************************
**  SENTIMENT ANALYSIS USING LSTM (3-CLASS SOFTMAX MODEL)  **
************************************************************

**OVERVIEW**
This project implements a Sentiment Analysis system using a
Long Short-Term Memory (LSTM) neural network to classify text
into three sentiment categories:

**• Positive**
**• Negative**
**• Rest (Neutral + Irrelevant)**

The model is trained on a Twitter sentiment dataset and built
using TensorFlow/Keras. It is suitable for college submission
and GitHub portfolio presentation.

------------------------------------------------------------

**PROJECT STRUCTURE**

demo-lstm/
│
├── main.py
│   → Model training and prediction script
├── data.csv / twitter_training.csv
│   → Dataset file (without column headers)
├── requirements.txt
│   → Project dependencies
└── README.md
│   → Project documentation

------------------------------------------------------------

**DATASET DESCRIPTION**

The dataset is a CSV file without column headers.
Columns are assigned manually during preprocessing.

**COLUMN MAPPING**
0 → id        : Unique identifier (not used)
1 → topic     : Topic or category
2 → sentiment : Sentiment label
3 → text      : Tweet / review content

**ORIGINAL SENTIMENT LABELS**
• Positive
• Negative
• Neutral
• Irrelevant

**FINAL SENTIMENT CLASSES USED**
• Positive
• Negative
• Rest (Neutral + Irrelevant)

------------------------------------------------------------

**TECHNOLOGIES USED**

• Python 3.10+
• TensorFlow / Keras
• Pandas
• NumPy
• Scikit-learn

------------------------------------------------------------

**DATA PREPROCESSING STEPS**

1. Load CSV file without headers
2. Assign column names manually
3. Remove missing or empty text rows
4. Convert text column to string format
5. Merge sentiment labels into 3 classes
6. Apply One-Hot Encoding
7. Tokenize text and apply padding

------------------------------------------------------------

**MODEL ARCHITECTURE**

Embedding Layer (vocab_size = 5000, output_dim = 128)
↓
LSTM Layer (128 units, dropout = 0.2)
↓
Dense Layer (3 units, Softmax)

**Training Configuration**
• Activation Function : Softmax
• Loss Function       : Categorical Crossentropy
• Optimizer           : Adam

------------------------------------------------------------

**HOW TO RUN THE PROJECT**

Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Run the Model
python main.py

------------------------------------------------------------

**EXAMPLE PREDICTIONS**

Text: This game is absolutely amazing
Prediction: Positive

Text: Worst experience ever
Prediction: Negative

Text: Just another normal day
Prediction: Rest

------------------------------------------------------------

**KEY FEATURES**

• Handles noisy Twitter text data
• Multi-class classification using Softmax
• Fully local execution
• Clean and modular codebase
• Suitable for academic and portfolio use

------------------------------------------------------------

**FUTURE IMPROVEMENTS**

• Implement Bi-LSTM for better context capture
• Handle class imbalance
• Add confusion matrix and classification report
• Save and load trained model
• Advanced text preprocessing

************************************************************
