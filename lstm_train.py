
# Emotion Analysis using LSTM Model

## 1. Importing Necessary Libraries

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import os
import json


model_path = 'model/emotion_model.h5'



def train_model():

    # Load the dataset
    df = pd.read_csv('data/train.txt', names=['text', 'label'], sep=';')

    # Displaying dataset info
    print('Training Dataset:')
    print(df.head())
    print(df['label'].value_counts())

    ## 3. Preprocessing the Data

    # Tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text'])

    # Save the tokenizer for future use
    with open('model/tokenizer.json', 'w') as f:
        f.write(tokenizer.to_json())

    # Convert text to sequences
    X_train = tokenizer.texts_to_sequences(df['text'])

    # Padding sequences to ensure uniform input size
    max_length = 50
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')

    # Label encoding
    label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}


    with open('model/label_to_index.json', 'w') as f:
        json.dump(label_to_index, f)

    y_train = np.array([label_to_index[label] for label in df['label']])

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

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=64,
        verbose=1
    )
    # Save the trained model
    model.save(model_path)
    print('Model saved to disk.')


if not os.path.exists(model_path):
    train_model()

else:
    # Load the saved model
    print('Model already exists')

