import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./fine_tuned_model"
TEST_PROMPT = "What is my age?"

def main():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        # Ensure the pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # If pad_token_id is a boolean, convert it to the correct ID
        if isinstance(tokenizer.pad_token_id, bool):
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, use_fast=False)
    except Exception as e:
        print("Error loading the fine-tuned model. Make sure fine_tuning.py has been run successfully.")
        print(str(e))
        return

    # Now tokenizer.encode should work as expected.
    input_ids = tokenizer.encode(TEST_PROMPT, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=150, num_return_sequences=1)
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Test Query Output:")
    print(output_text)

if __name__ == "__main__":
    main()
