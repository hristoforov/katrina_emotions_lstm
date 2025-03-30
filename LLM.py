from llama_cpp import Llama
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import re

def setup_model():
    # Load the GGUF model file (make sure to provide the correct path to your model file)
    #model_path = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    # Initialize the model
    #return Llama(model_path=model_path, verbose=False)
    return Llama.from_pretrained(
        repo_id="lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q3_K_L.gguf",
        verbose=False
    )

def load_data(file_path):
    """Load and preprocess the emotion dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, emotion = line.strip().split(';')
            data.append({'text': text, 'emotion': emotion})
    return pd.DataFrame(data)

def cleanup(text):
    valid_sentiments = {"joy", "sadness", "anger", "fear", "love", "surprise"}
    words = re.findall(r'\b\w+\b', text)
    if (len(words) > 0) & (words[0] in valid_sentiments):
        return words[0]
    else:
        return "Unknown"

def analyze(model, text):


    prompt = f"""<|start_header_id|>system<|end_header_id|>
    Analyze the sentiment of the given text, including the emotions expressed. Answer with exactly one of: joy, sadness, anger, fear, love, or surprise.
    
    <|start_header_id|>user<|end_header_id|>
    {text}
    
    <|start_header_id|>assistant<|end_header_id|>
    """

    response = model(prompt, max_tokens=100, stop=["<|start_header_id|>"])
    return response["choices"][0]["text"].strip().lower()




def evaluate_model(model, test_data):
    predictions = []
    true_labels = []

    print("\nEvaluating model...\n")
    skipped = 0
    for _, row in test_data.iterrows():

        raw_emotion = analyze(model, row['text'])
        predicted_emotion = cleanup(raw_emotion)
        print(f"Text: {row['text']}")
        print(f"True emotion: {row['emotion']}")

        if predicted_emotion != "Unknown":
            predictions.append(predicted_emotion)

        else:
            skipped = skipped + 1

        print(f"True: {row['emotion']} , Predicted : {raw_emotion}\n")


    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    print("\nSkipped "+skipped)
def main():
    # Load data
    print("Loading data...")
    df =  load_data('data/test.txt')

    # Set up model
    print("Setting up model...")
    model = setup_model()

    # Use more samples for testing (50 samples)
    test_data = df #.head(50)  # Use first 50 samples after shuffling
    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()