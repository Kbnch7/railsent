import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np

def load_model_and_tokenizer(model_path, tokenizer_name):
    model = tf.keras.models.load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

def predict_sentiment(texts, model, tokenizer):
    encoded_texts = tokenizer(texts, padding='max_length', truncation=True, max_length=200, return_tensors="np")
    
    input_ids = encoded_texts['input_ids']

    # Predict
    predictions = model.predict(input_ids)
    
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels - 1

def test_with_known_texts(model, tokenizer):
    texts = [
        "Супер круто",
        "Чувак, вечеринка просто отстой.",
        "11й маршрут, на станции Войковская сделайте выдержку 1 минуту."
    ]
    labels = ["Positive", "Negative", "Neutral"]
    predictions = predict_sentiment(texts, model, tokenizer)
    for text, predicted_label in zip(texts, predictions):
        print(f"Text: {text}\nPredicted sentiment: {labels[predicted_label + 1]}\n")


def main():
    print("Loading model and tokenizer...")
    model_path = 'gru_sentiment_model'
    tokenizer_name = "DeepPavlov/rubert-base-cased"
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_name)
    print("Model and tokenizer loaded successfully.")
    test_with_known_texts(model, tokenizer)


if __name__ == "__main__":
    main()
