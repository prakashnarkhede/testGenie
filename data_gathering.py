import os
import json
import re

try:
    from docx import Document
except ImportError:
    print("python-docx is not installed. Please install it via 'pip install python-docx'.")

def read_txt_file(filepath):
    """
    Reads a TXT file and returns its content as a string.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def read_docx_file(filepath):
    """
    Reads a DOCX file and returns its content as a string.
    """
    doc = Document(filepath)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def parse_training_data(raw_text):
    """
    Parses raw text into a list of samples.
    Each sample is a block of text (raw information) about the application.
    Samples are separated by a line with three or more dashes.
    """
    samples = []
    # Split based on a line with at least three dashes.
    raw_samples = re.split(r'\n\s*-{3,}\s*\n', raw_text.strip())
    
    for sample in raw_samples:
        text = sample.strip()
        if text:
            samples.append({"info": text})
    return samples

def gather_training_data():
    """
    Reads training data from either a TXT file or a DOCX file (if TXT isn't found),
    parses the text into samples, and saves the structured samples as JSON.
    """
    training_file = ""
    if os.path.exists("./TrainingData/training_data.txt"):
        training_file = "./TrainingData/training_data.txt"
        print("Found training_data.txt. Reading from TXT file.")
        raw_text = read_txt_file(training_file)
    elif os.path.exists("./TrainingData/training_data.docx"):
        training_file = "./TrainingData/training_data.docx"
        print("Found training_data.docx. Reading from DOCX file.")
        raw_text = read_docx_file(training_file)
    else:
        print("No training data file found. Please provide training_data.txt or training_data.docx in the project directory.")
        return

    samples = parse_training_data(raw_text)
    if not samples:
        print("No valid samples found in the training data file.")
        return

    with open("extracted_data.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)
    print(f"Extracted {len(samples)} samples and saved to extracted_data.json.")

if __name__ == "__main__":
    gather_training_data()
