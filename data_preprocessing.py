import json
import re

def preprocess_text(text):
    """
    Cleans the input text by:
    - Converting it to lowercase.
    - Removing punctuation.
    - Stripping extra whitespace.
    """
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def preprocess_data():
    """
    Reads extracted data from extracted_data.json, processes each sample,
    and saves the result to processed_data.json.
    """
    try:
        with open("extracted_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("extracted_data.json not found. Please run data_gathering.py first.")
        return

    processed_data = []
    for item in data:
        processed_item = {
            "info": preprocess_text(item["info"])
        }
        processed_data.append(processed_item)

    with open("processed_data.json", "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)
    print("Data preprocessing complete. Processed data saved to processed_data.json.")

if __name__ == "__main__":
    preprocess_data()
