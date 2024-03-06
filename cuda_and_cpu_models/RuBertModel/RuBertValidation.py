import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np

def load_model_and_tokenizer(model_path, tokenizer_name):
    model = tf.keras.models.load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

def preprocess_and_encode_texts(texts, tokenizer):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="tf")
    return dict(encoded)


def predict_sentiment(texts, model, tokenizer):
    # Preprocess and encode texts
    encoded_inputs = preprocess_and_encode_texts(texts, tokenizer)
    # Predict
    predictions = model.predict(encoded_inputs)
    if 'logits' in predictions:
        logits = predictions['logits']
    else:
        logits = predictions
    predicted_labels = np.argmax(logits, axis=1)
    sentiment_labels = ['neutral', 'positive', 'negative']
    predicted_sentiments = [sentiment_labels[label] for label in predicted_labels]
    return predicted_sentiments

def main():
    model_path = '/app/rubert_sentiment_model'
    tokenizer_name = 'DeepPavlov/rubert-base-cased'
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_name)
    
    texts = [
        "Щенок, гоняющийся за собственным хвостом, - это самое милое, самое веселое, что я видел за весь день!"
    ]
    
    predictions = predict_sentiment(texts, model, tokenizer)

    for text, prediction in zip(texts, predictions):
        print(f"'{text}' - {prediction}")

if __name__ == "__main__":
    main()
