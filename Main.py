
# Emotion Analysis using LSTM Model

## 1. Importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

## 2. Loading the Dataset

# Load the dataset
df = pd.read_csv('data/train.txt', names=['text', 'label'], sep=';')
df_test = pd.read_csv('data/test.txt', names=['text', 'label'], sep=';')

# Displaying dataset info
print('Training Dataset:')
print(df.head())
print(df['label'].value_counts())
print()
print('Test Dataset:')
print(df_test.head())
print(df_test['label'].value_counts())

## 3. Preprocessing the Data

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

# Convert text to sequences
X_train = tokenizer.texts_to_sequences(df['text'])
X_test = tokenizer.texts_to_sequences(df_test['text'])

# Padding sequences to ensure uniform input size
max_length = 50
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

# Label encoding
label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
index_to_label = {idx: label for label, idx in label_to_index.items()}

y_train = np.array([label_to_index[label] for label in df['label']])
y_test = np.array([label_to_index[label] for label in df_test['label']])

## 4. Building the LSTM Model

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(len(label_to_index), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## 5. Training the Model

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=1
)

## 6. Evaluating the Model

y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=list(label_to_index.keys())))

## 7. Making Predictions

def predict_emotion(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = np.argmax(model.predict(padded_sequence), axis=1)[0]
    return index_to_label[prediction]

# Example prediction
print(predict_emotion('I am so happy today!'))

