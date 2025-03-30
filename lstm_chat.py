# chat_local_model.py

import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Emotion to emoji mapping
EMOTION_EMOJI = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😮',
    'love': '❤️',
    'neutral': '😐',
    'unknown': '❓'
}

# 1. Load Model and Tokenizer
model = load_model('model/emotion_model.h5')

with open('model/tokenizer.json', 'r') as f:
    json_string = f.read()
    tokenizer = tokenizer_from_json(json_string)

with open('model/label_to_index.json', 'r') as f:
    label_to_index = json.load(f)

# Rebuild index_to_label
index_to_label = {idx: label for label, idx in label_to_index.items()}
index_to_label = {int(k): v for k, v in index_to_label.items()}

# 2. Prediction function
def predict_emotion(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post')
    prediction = np.argmax(model.predict(padded_sequence), axis=1)[0]
    emotion = index_to_label.get(prediction, "unknown")
    return emotion

# 3. Main Chat Loop
def main():
    print("Emotion Chat (Local Model) - Type 'quit' to exit")
    print("I'll analyze your text and respond with an emoji representing the emotion.")
    print("Available emotions: joy, sadness, anger, fear, surprise, love, neutral")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        emotion = predict_emotion(user_input)
        emoji = EMOTION_EMOJI.get(emotion, EMOTION_EMOJI['unknown'])

        print(f"Bot: {emoji} ({emotion})")

if __name__ == "__main__":
    main()