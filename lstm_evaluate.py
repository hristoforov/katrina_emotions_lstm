# evaluate.py

# 1. Import Libraries
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

# 2. Load Test Dataset
df_test = pd.read_csv('data/test.txt', names=['text', 'label'], sep=';')

# 3. Load Tokenizer
with open('model/tokenizer.json', 'r') as f:
    json_string = f.read()  # read it as a string
    tokenizer = tokenizer_from_json(json_string)  # pass string, not dict

X_test = tokenizer.texts_to_sequences(df_test['text'])
X_test = pad_sequences(X_test, maxlen=50, padding='post')

# Label encoding (should match training encoding)
# Load label_to_index
with open('model/label_to_index.json', 'r') as f:
    label_to_index = json.load(f)

# Rebuild index_to_label
index_to_label = {idx: label for label, idx in label_to_index.items()}

y_test = np.array([label_to_index[label] for label in df_test['label']])

# 4. Load Model
model = load_model('model/emotion_model.h5')

# 5. Evaluate Model
y_pred = np.argmax(model.predict(X_test), axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=list(label_to_index.keys())))

# 6. Predict New Emotion
def predict_emotion(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post')
    prediction = np.argmax(model.predict(padded_sequence), axis=1)[0]
    return index_to_label[prediction]

# Example
print(predict_emotion('I am so happy today!'))