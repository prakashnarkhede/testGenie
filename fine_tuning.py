import json
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Use the meta Llama model (or your chosen model)
MODEL_NAME = "meta-llama/Llama-3.2-1B"

def load_processed_data():
    try:
        with open("processed_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("processed_data.json not found. Please run data_preprocessing.py first.")
        return None

    dataset = []
    for item in data:
        prompt = f"Information: {item['info']}\nTest Case:"
        dataset.append({"input_text": prompt})
    return dataset

def tokenize_function(examples, tokenizer, max_length=256):
    inputs = examples["input_text"]
    encoding = tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")
    # Set the labels to be the same as the input IDs
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

def main():
    data = load_processed_data()
    if data is None:
        return

    dataset = Dataset.from_list(data)
    
    # Load tokenizer with trust_remote_code enabled
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if isinstance(tokenizer.pad_token_id, bool):
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuning complete. Model and tokenizer saved to ./fine_tuned_model.")

if __name__ == "__main__":
    main()
