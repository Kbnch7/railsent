import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
import numpy as np

def load_model(model_bin_path, model_name="sberbank-ai/rugpt3small_based_on_gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(model_name, num_labels=3)
    model = GPT2ForSequenceClassification(config)
    model.load_state_dict(torch.load(model_bin_path, map_location=torch.device('cpu')))
    return model, tokenizer

def preprocess_text(text):
    text = text.lower()
    return text

def predict_sentiment(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    processed_text = preprocess_text(text)
    inputs = tokenizer.encode(processed_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
    return predictions

def main():
    model_path = '/app/models/rugpt3small_based_on_gpt2.bin'
    model, tokenizer = load_model(model_path)

    texts = [

"Мы всегда готовы помочь вам и предложить вам лучший сервис на протяжении всего пути ваш комфорт для нас очень важен!"
    ]

    sentiment_labels = ['Neutral', 'Positive', 'Negative']

    for text in texts:
        predictions = predict_sentiment(text, model, tokenizer)
        predicted_sentiment = sentiment_labels[np.argmax(predictions.cpu().numpy())]
        print(f"Text: {text}\nPredicted sentiment: {predicted_sentiment}\n")

if __name__ == "__main__":
    main()