# 🎭 Shakespeare Next Word Predictor (Streamlit App)

A simple **Machine Learning web app** that predicts the next word in a sentence using an **LSTM model** trained on Shakespeare’s *Hamlet*.

---

## 🚀 Features

- 🌐 Interactive web app using Streamlit  
- 🧠 LSTM and GRU based next word prediction model  
- 🎭 Trained on Shakespeare’s *Hamlet* text  
- ⚡ Real-time predictions in browser  
- 📦 Uses saved tokenizer + trained model  

---

## 🤔 How It Works

1. User enters a sentence in the input box  
2. Text is converted into sequences using tokenizer  
3. Sequence is padded to required length  
4. LSTM model predicts probability of next words  
5. One word is selected from the probability distribution  
6. Result is displayed on the screen  

---

## 🖥️ Web App Code (Streamlit)

```python
import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model('lstm_model.keras')

st.title("🎭 Shakespeare Next Word Predictor")

text = st.text_input("Enter your text")

if st.button("Predict Next Word"):

    # Convert text to sequence
    input_tokenize = tokenizer.texts_to_sequences([text])

    # Pad sequence
    input_pad = pad_sequences(input_tokenize, maxlen=13, padding='pre')

    # Predict
    preds = model.predict(input_pad, verbose=0)[0]

    # Pick index based on probability
    pos = np.random.choice(len(preds), p=preds)

    # Convert index to word
    word = ""
    for w, index in tokenizer.word_index.items():
        if index == pos:
            word = w
            break

    st.success(f"Predicted next word: {word}")

```
## 📁 Project Structure
```
shakespeare-next-word-predictor/
│
├── app.py                # Streamlit app
├── lstm_model.keras      # Trained LSTM model
├── tokenizer.pkl        # Saved tokenizer
├── requirements.txt     # Dependencies
└── README.md            # Documentation

```
## 🧠 Model Details
Model Type: LSTM (Recurrent Neural Network) and GRU
Dataset: Shakespeare’s Hamlet
Tokenizer: Keras Tokenizer
Input: Padded word sequences
Output: Probability of next word

## ⚠️ Limitations
Predicts only one word at a time
Trained only on Hamlet
Output may vary due to randomness
Works best for Shakespeare-style text

## 🚀 Future Improvements
Generate full sentences (loop predictions)
Train on full Shakespeare corpus
Replace LSTM with Transformer (GPT-style model)
Show top 3 predicted words instead of one
Deploy on Streamlit Cloud
