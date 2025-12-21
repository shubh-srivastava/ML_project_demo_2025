🧠 Sentiment Analysis using LSTM (Softmax – 3 Classes)
This project implements a Sentiment Analysis model using LSTM (Long Short-Term Memory) neural networks to classify text data into three sentiment categories:

Positive

Negative

Rest (Neutral, Irrelevant, or any other sentiment)

The model is trained on a Twitter sentiment dataset and uses deep learning with TensorFlow/Keras.

📂 Project Structure
demo-lstm/
│
├── main.py              # Main Python script (model training & prediction)
├── data.csv / twitter_training.csv   # Dataset file (no column headers)
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
📊 Dataset Description
The dataset is a CSV file without column names

Columns are assigned manually as:

Column Index	Column Name	Description
0	id	Unique identifier (not used)
1	topic	Topic or category
2	sentiment	Sentiment label
3	text	Actual tweet / review text
Original Sentiment Labels
Positive

Negative

Neutral

Irrelevant

Final Sentiment Classes (Used in Model)
Positive

Negative

Rest (Neutral + Irrelevant)

⚙️ Technologies Used
Python 3.10+

TensorFlow / Keras

Pandas

NumPy

Scikit-learn

🧪 Data Preprocessing Steps
Load CSV without headers

Assign column names manually

Remove missing or empty text rows

Convert text column to string

Group sentiments into 3 classes

Apply One‑Hot Encoding

Tokenize text and apply padding

🧠 Model Architecture
Embedding Layer (5000 vocab, 128 dim)
↓
LSTM Layer (128 units, dropout=0.2)
↓
Dense Layer (3 units, softmax)
Activation Function: Softmax

Loss Function: Categorical Crossentropy

Optimizer: Adam

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run the Model
python main.py
🔍 Example Predictions
Text: This game is absolutely amazing
Predicted Sentiment: Positive

Text: Worst experience ever
Predicted Sentiment: Negative

Text: Just another normal day
Predicted Sentiment: Rest
✅ Key Features
Handles missing & noisy Twitter data

Uses Softmax for multi‑class classification

Fully local execution (no Kaggle / Colab dependency)

Clean, modular, and readable code

Ready for college submission or GitHub portfolio

📌 Future Improvements
Add Bi‑LSTM for better context understanding

Handle class imbalance

Add confusion matrix & classification report

Save and load trained model

Advanced text cleaning (URLs, emojis, hashtags)

