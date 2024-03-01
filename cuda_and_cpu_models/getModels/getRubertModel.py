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
    model_path = 'rubert_sentiment_model'
    tokenizer_name = 'DeepPavlov/rubert-base-cased'
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_name)
    
    texts = [
        "Светит солнце, поют птицы, и я никогда не чувствовал себя более живым, чем в этот прекрасный день!",
        "Я не могу поверить, что моя машина снова сломалась, как раз тогда, когда я уже опаздывал на важную встречу",
        'Годовой отчет был представлен вовремя со всеми необходимыми даннымин.',
        "О, здорово, еще одна встреча в понедельник утром, которая наполнит мой день безграничной радостью",
        "Я так взволнован запуском нашего нового продукта; я просто знаю, что он будет иметь огромный успех!",
        "Я смотрел новости о шторме, и я действительно беспокоюсь о людях, оказавшихся на его пути",
        "Полное разочарование. Первый раз со мной такое. Обычно не оставляю отзывы, здесь просто не могу промолчать. Записалась на аппаратный педикюр и маникюр с шеллаком к Наталии, это мой первый и последний раз посещения этого заведения",
        "Стоимость не соответствует качеству. За миникюр (покрытие гель лак) и педикюр с обычным лаком отдала 4500. Маникюр сделать не качественно, есть затеки и не прокрашенные места. За такую цену явно качество должно быть выше.",
        "Отмечал в этом ужасном заведении День рождения .  Еда хорошая , интерьер приятный , но вот обслуживание - ОТВРАТИТЕЛЬНОЕ .  Одновременно с нами было еще 4 компании .  Нам блюда приносили минимум через 2 часа ( стол практически весь вечер был пустой , за исключением напитков )",
        "Щенок, гоняющийся за собственным хвостом, - это самое милое, самое веселое, что я видел за весь день!"
    ]
    
    predictions = predict_sentiment(texts, model, tokenizer)

    for text, prediction in zip(texts, predictions):
        print(f"'{text}' - {prediction}")

if __name__ == "__main__":
    main()
