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
    model_bin_path = "./models/rugpt3small_based_on_gpt2_accur80.bin"
    model, tokenizer = load_model(model_bin_path)

    texts = [

"Мы всегда готовы помочь вам и предложить вам лучший сервис на протяжении всего пути ваш комфорт для нас очень важен!,1",
"Спасибо за ваше доверие и преданность нашей компании!,1",
"Благодарим за ваше доброе сердце и любовь к путешествиям ваше участие делает наши поездки более захватывающими и волшебными!,1",
"Уважаемые коллеги давайте проявим заботу и внимание к каждому пассажиру чтобы сделать их путешествие особенным.,1",
"Мы гордимся тем что обеспечиваем ваше перемещение с комфортом и вовремя.,1",
"Наша цель - сделать ваше путешествие комфортным и безопасным.,1",

"Номер хвостового вагона соответствует натурному листу.,0",
"На вас электропоезд едет.,0",
"Диспетчер 31-я минута 61-я на третьем станционе 55-й отправился тоже 31-20.,0",
"Расценки завышены.,0",
"61-й надо завести вам лично на третий станционный путь.,0",
"Зеленые стрелки в маршруте по первому главному пути.,0",
"Страшное документальное свидетельство последних минут перед катастрофой.,0",
"Указание к нам пришло на Пензу 4 вас сгнать Ушаков.,0",
"Машинист михайлов 4-4 вижу м13 белый выполняю доведенный план мировой работы работаем по сигналу.,0",
"Мне-то вот это вопрос.,0"


"?распускают руки и бьют людей!!!?,2",
"Ваши работники перешли все границы!!,2",
"Рассматриваю варианты тольятти (самара) - санкт -петербург.,2",
"Можно с пересадкой через москву.,2",
"С самого начало все пошло не так!,2",
"Уже на входе.,2",
"У каждого паспорт проверяли очень долго очередь была метров 5!,2",
"Мало того хамят так ещё и поез проветрить неудосужились!,"
    ]

    sentiment_labels = ['Neutral', 'Positive', 'Negative']

    for text in texts:
        predictions = predict_sentiment(text, model, tokenizer)
        predicted_sentiment = sentiment_labels[np.argmax(predictions.cpu().numpy())]
        print(f"Text: {text}\nPredicted sentiment: {predicted_sentiment}\n")

if __name__ == "__main__":
    main()