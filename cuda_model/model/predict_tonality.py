import numpy as np
import fasttext
from joblib import load
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

nltk.download('stopwords', quiet=True)

model = load('trained_model.joblib')

ft_model = fasttext.load_model('cc.ru.300.bin')

stop_words = stopwords.words('russian')
stemmer = SnowballStemmer('russian')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


def vectorize_with_fasttext(text, model):
    words = text.split()
    vectors = np.array([model.get_word_vector(word) for word in words])
    if len(vectors) == 0:
        return np.zeros(model.get_dimension())
    else:
        return np.mean(vectors, axis=0)


def predict_tonality(sentence):
    processed_text = preprocess_text(sentence)
    vectorized_text = vectorize_with_fasttext(processed_text, ft_model).reshape(1, -1)
    prediction = model.predict(vectorized_text)
    return prediction[0]


if __name__ == "__main__":
    sentences = [
        "отстой.",
        "11й маршрут, на станции Войковская сделайте выдержку 1 минуту.",
        "Чувак эта вечеринка просто улет."
    ]

    for sentence in sentences:
        tonality = predict_tonality(sentence)
        print(f"Sentence: \"{sentence}\" \nPredicted Tonality: {tonality}\n")
