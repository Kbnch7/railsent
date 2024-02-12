import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# def configure_gpu():
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             print(f"GPUs available: {gpus}")
#         except RuntimeError as e:
#             print(e)
#     else:
#         print("No GPU, using CPU.")

def configure_cpu():
    tf.config.set_visible_devices([], 'GPU')
    print("Configured to use CPU.")

def download_nltk_data():
    nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    stop_words = stopwords.words('russian')
    stemmer = SnowballStemmer('russian')
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    tqdm.pandas(desc="Processing Texts")
    
    sample_texts = df.sample(5)['text']
    for text in sample_texts:
        print("Original text:", text)
        print("Processed text:", preprocess_text(text))
        print("---")

    df['processed_text'] = df['text'].progress_apply(preprocess_text)

    return df


class CustomGRUModel(Model):
    def __init__(self, vocab_size, embedding_dim, input_length, gru_units, dropout_rate, num_classes):
        super(CustomGRUModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)

        self.spatial_dropout = SpatialDropout1D(dropout_rate)
        self.bidirectional_gru = Bidirectional(GRU(gru_units, return_sequences=False))
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.spatial_dropout(x)
        x = self.bidirectional_gru(x)
        return self.dense(x)

def encode_texts(tokenizer, texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=200, return_tensors="np")

def get_misclassified_texts(X_test, y_test, predictions, tokenizer):
    """
    Identify and return misclassified texts along with actual & predicted labels
    """
    texts = []
    for seq in X_test:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        texts.append(text)
    
    predicted = np.argmax(predictions, axis=1)
    
    y_test_adjusted = y_test
    misclassified = y_test_adjusted != predicted
    misclassified_texts = pd.DataFrame({
        'text': np.array(texts)[misclassified],
        'actual_label': y_test_adjusted[misclassified],
        'predicted_label': predicted[misclassified]
    })
    
    return misclassified_texts

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = CustomGRUModel(vocab_size=119547,
                        embedding_dim=128, 
                        input_length=200, 
                        gru_units=64, 
                        dropout_rate=0.2, 
                        num_classes=3)
    model.build(input_shape=(None, 200))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[early_stopping], verbose=2)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test Accuracy: {accuracy}')


    # Generate predictions
    predictions = model.predict(X_test, verbose=2)
    predicted_labels = np.argmax(predictions, axis=1) 
    print("Unique values in y_test:", np.unique(y_test))
    print("Unique values in predicted_labels:", np.unique(predicted_labels))

    print("Classification Report:")
    print(classification_report(y_test, predicted_labels, labels=[0, 1, 2], target_names=['Neutral', 'Positive', 'Negative'], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, predicted_labels, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Positive', 'Negative'], yticklabels=['Neutral', 'Positive', 'Negative'])

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return model, y_test, predictions


def main():
    # configure_gpu()
    configure_cpu()
    download_nltk_data()
    
    train_filepath = 'train.csv'
    train_df = load_and_preprocess_data(train_filepath)
    print(train_df[['sentiment', 'text']].head())

    labels = train_df['sentiment'].values

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    vocab_size = len(tokenizer.get_vocab())
    print(f"Tokenizer vocab size: {vocab_size}")

    encodings = tokenizer(train_df['processed_text'].tolist(), truncation=True, padding=True, max_length=200, return_tensors="np")
    X = encodings['input_ids']
    y = train_df['sentiment'].values
    print(np.unique(labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, y_test, predictions = train_and_evaluate_model(X_train, y_train, X_test, y_test)


    predicted_labels = np.argmax(predictions, axis=1)
    
    y_test_adjusted = y_test
    misclassified_texts = get_misclassified_texts(X_test, y_test, predictions, tokenizer)
    print("Misclassified texts sample:")
    print(misclassified_texts.sample(10))
    model.save('gru_sentiment_model', save_format='tf')
    loaded_model = tf.keras.models.load_model('gru_sentiment_model')

    sample_text = train_df['processed_text'].iloc[0]
    encoded_text = encode_texts(tokenizer, [sample_text])
    prediction = loaded_model.predict(encoded_text['input_ids'])

    print(prediction)
    cm = confusion_matrix(y_test_adjusted, predicted_labels)
    print("Confusion Matrix:")
    print(cm)
    
    print("Classification Report:")
    print(classification_report(y_test_adjusted, predicted_labels))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()