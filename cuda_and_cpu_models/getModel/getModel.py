import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer
import re
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
import pymorphy2

def download_nltk_data():
    nltk.download('stopwords', quiet=True)

morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    stop_words = stopwords.words('russian')
    words = [word for word in words if word not in stop_words]
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

model_path = 'cnn_gru_sentiment_model_optimized'
model = load_model(model_path)

def encode_texts(texts):
    encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=200, return_tensors="np")
    return encoded['input_ids']

def classify_texts(texts):
    preprocessed_texts = [preprocess_text(text) for text in texts]
    encoded_texts = encode_texts(preprocessed_texts)
    predictions = model.predict(encoded_texts)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

def activate_first_key(predicted_labels):
    for label in predicted_labels:
        if label == 0:
            print("Activated for NEUTRAL")
        elif label == 1:
            print("Activated for POSITIVE")
        elif label == 2:
            print("Activated for NEGATIVE")
        else:
            print("Unknown label")

texts = [
    "отличный, милый хорошой, лучший человек",
    "11й маршрут, на станции Войковская сделайте выдержку 1 минуту.",
    "Я вас ненавижу!",
    "FFFFFFFFFF"
]
predicted_labels = classify_texts(texts)
activate_first_key(predicted_labels)
