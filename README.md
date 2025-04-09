# Llama-Powered Test Case Generation System

This project demonstrates an end-to-end AI solution using:
- **Streamlit** for an interactive user interface.
- **LangChain** to orchestrate prompt workflows.
- **A fine-tuned Llama model** for generating test cases.
- **Data Ingestion from TXT or DOCX documents** for training data.

## Project Structure
- **README.md**: Overview, setup instructions, and project structure.
- **requirements.txt**: List of required Python packages.
- **data_gathering.py**: Reads training data from a TXT file or DOCX file and saves it in JSON format.
- **data_preprocessing.py**: Cleans and prepares the extracted data.
- **fine_tuning.py**: Fine-tunes the chosen Llama model on the preprocessed dataset.
- **local_testing.py**: Loads the fine-tuned model and runs a sample query.
- **app.py**: A Streamlit app that uses LangChain to pass user queries to the model interactively.

## Environment & Setup Instructions

1. **Install dependencies:**  
   Ensure you have Python 3.8+ installed. Then run:
   ```bash
   pip install -r requirements.txt

activate venv:
.venv\Scripts\activate


huggingface login via cli:   huggingface-cli login

use token: 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


python data_gathering.py
python data_preprocessing.py
python fine_tuning.py
python local_testing.py
streamlit run app.py

