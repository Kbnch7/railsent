import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, Dense, Input, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from transformers import AutoTokenizer
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
import pymorphy2

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

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    tqdm.pandas(desc="Processing Texts")
    
    df['processed_text'] = df['text'].progress_apply(preprocess_text)
    
    samples = df.sample(2)
    for index, row in samples.iterrows():
        print("Original text:", row['text'])
        print("Processed text:", row['processed_text'])
        print("---")

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

def create_cnn_gru_model(vocab_size, embedding_dim, input_length, gru_units, dropout_rate, num_classes, filters, kernel_size):
    input_layer = Input(shape=(input_length,))
    
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(input_layer)
    x = SpatialDropout1D(dropout_rate)(x)
    
    conv_layer = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(x)
    conv_layer = GlobalMaxPooling1D()(conv_layer)
    
    gru_layer = Bidirectional(GRU(gru_units, return_sequences=False))(x)
    
    concatenated = Concatenate()([conv_layer, gru_layer])
    
    output_layer = Dense(num_classes, activation='softmax')(concatenated)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def train_and_evaluate_model(X_train, y_train, X_val, y_val, vocab_size=119547, embedding_dim=128, input_length=200, gru_units=128, dropout_rate=0.2, num_classes=3, filters=64, kernel_size=3, learning_rate=0.001):
    model = create_cnn_gru_model(vocab_size=vocab_size, embedding_dim=embedding_dim, input_length=input_length, gru_units=gru_units, dropout_rate=dropout_rate, num_classes=num_classes, filters=filters, kernel_size=kernel_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, callbacks=[early_stopping], verbose=2)

    loss, accuracy = model.evaluate(X_val, y_val, verbose=2)
    print(f'Test Accuracy: {accuracy}')

    predictions = model.predict(X_val, verbose=2)
    predicted_labels = np.argmax(predictions, axis=1)
    print("Unique values in y_test:", np.unique(y_val))
    print("Unique values in predicted_labels:", np.unique(predicted_labels))

    print("Classification Report:")
    print(classification_report(y_val, predicted_labels, labels=[0, 1, 2], target_names=['Neutral', 'Positive', 'Negative'], zero_division=0))

    cm = confusion_matrix(y_val, predicted_labels, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Positive', 'Negative'], yticklabels=['Neutral', 'Positive', 'Negative'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return model, accuracy, predictions

def main():
    configure_cpu()
    # configure_gpu()
    download_nltk_data()
    
    train_filepath = 'train.csv'
    train_df = load_and_preprocess_data(train_filepath)
    
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    vocab_size = len(tokenizer.get_vocab())
    print(f"Tokenizer vocab size: {vocab_size}")

    train_encodings = tokenizer(train_df['processed_text'].tolist(), truncation=True, padding=True, max_length=200, return_tensors="np")
    X_train = train_encodings['input_ids']
    y_train = train_df['sentiment'].values
    
    validation_filepath = 'valid.csv'
    validation_df = load_and_preprocess_data(validation_filepath)
    
    validation_encodings = tokenizer(validation_df['processed_text'].tolist(), truncation=True, padding=True, max_length=200, return_tensors="np")
    X_val = validation_encodings['input_ids']
    y_val = validation_df['sentiment'].values
    
    def objective(trial):
        gru_units = trial.suggest_categorical('gru_units', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        filters = trial.suggest_categorical('filters', [32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])

        _, accuracy, _ = train_and_evaluate_model(X_train, y_train, X_val, y_val, gru_units=gru_units, dropout_rate=dropout_rate, learning_rate=learning_rate, filters=filters, kernel_size=kernel_size)
        return accuracy


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)

    best_trial = study.best_trial
    print("Best trial:")
    print(f"Value: {best_trial.value}")
    print("Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    best_params = best_trial.params
    model, accuracy, predictions = train_and_evaluate_model(
        X_train, y_train, X_val, y_val,
        gru_units=best_params['gru_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        filters=best_params.get('filters', 64), 
        kernel_size=best_params.get('kernel_size', 3) 
    )

    predicted_labels = np.argmax(predictions, axis=1)
    print("Classification Report for Validation Dataset:")
    print(classification_report(y_val, predicted_labels, target_names=['Negative', 'Neutral', 'Positive']))

    misclassified_texts = get_misclassified_texts(X_val, y_val, predictions, tokenizer)
    print("Sample of misclassified texts:")
    print(misclassified_texts.sample(min(10, len(misclassified_texts))))

    try:
        model.save('cnn_gru_sentiment_model_optimized', save_format='tf')
    except Exception as e:
        print("Model save failed:", e)

if __name__ == "__main__":
    main()