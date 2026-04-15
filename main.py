import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer.pkl','rb')as f:
    tokenizer=pickle.load(f)

model=tf.keras.models.load_model('lstm_model.keras')

st.title("Next Word Predictor")

text = st.text_input("Enter your text")

if st.button("pedict next word"):
    
    #tokenize
    input_tokenize=tokenizer.texts_to_sequences([text])

    # padded input
    input_pad=pad_sequences(input_tokenize,maxlen=14-1,padding='pre')

    #predict
    preds=model.predict(input_pad)[0]
    pos=np.random.choice(len(preds),p=preds)

    for word,index in tokenizer.word_index.items():
        if index==pos:
            print(word)
            st.write('predicted word:',word)