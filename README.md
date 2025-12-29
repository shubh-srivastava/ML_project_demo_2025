# 🎵 Music Genre Classification using CNN

This project implements a **Convolutional Neural Network (CNN)**–based deep learning model to automatically classify music tracks into different genres using audio features.

## 🚀 Overview
- Converts audio files into **Mel Spectrograms**
- Uses a **CNN** to learn spatial audio patterns
- Predicts the **music genre** of a given track

## 🧠 Model
- Input: Mel Spectrograms extracted from audio files  
- Architecture: Convolutional layers + pooling + dense layers  
- Output: Genre probability distribution  

## 📂 Dataset
- Public music genre dataset (e.g., GTZAN)  
- Audio files labeled by genre  

## ⚙️ Tech Stack
- Python  
- TensorFlow / Keras  
- Librosa  
- NumPy, Matplotlib  

## 📈 Results
- Achieves strong genre classification accuracy  
- Visualized training and validation performance  

## 🏁 How to Run
```bash
pip install -r requirements.txt
python train.py
