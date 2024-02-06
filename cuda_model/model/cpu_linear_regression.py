import pandas as pd
import re
import nltk
import numpy as np
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump, load
import os

nltk.download('stopwords', quiet=True)

try:
    ft_model = fasttext.load_model('cc.ru.300.bin')
except Exception as e:
    print(f"Error loading FastText model: {e}")
    exit()

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
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

    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

def vectorize_with_fasttext(text, model):
    words = text.split()
    vectors = np.array([model.get_word_vector(word) for word in words])
    if len(vectors) == 0:
        return np.zeros(model.get_dimension())
    else:
        return np.mean(vectors, axis=0)

def feature_extraction(df, model):
    vectors = np.array([vectorize_with_fasttext(text, model) for text in df['processed_text']])
    y = df['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    return vectors, y

def train_model(X_train, y_train, model_path='trained_model.joblib'):
    if os.path.exists(model_path):
        model = load(model_path)
        print("Model loaded.")
    else:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(zip(np.unique(y_train), class_weights))
        model = LogisticRegression(class_weight=class_weights_dict, max_iter=1000)
        model.fit(X_train, y_train)
        dump(model, model_path)
        print("Model saved.")
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)

def main():
    try:
        filepath = 'cleaned_data.csv'
        df = load_and_preprocess_data(filepath)
        X, y = feature_extraction(df, ft_model)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        model_path = 'trained_model.joblib'
        model = train_model(X_train, y_train, model_path)
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
