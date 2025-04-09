import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# Path to the fine-tuned model directory.
MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_PATH = "./fine_tuned_model"  # Use this for loading fine-tuned model files

def load_model_pipeline():
    """
    Loads the fine-tuned Llama model and tokenizer,
    then returns a text generation pipeline wrapped by LangChain.
    """
    try:
      
      #  tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # Set the pad token to the eos_token
            tokenizer.pad_token = tokenizer.eos_token
        # If pad_token_id is still a bool, convert it to a proper ID
        if isinstance(tokenizer.pad_token_id, bool):
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=150,
            temperature=0.7,
        )
        return text_gen_pipeline
    except Exception as e:
        st.error("Failed to load model pipeline. Ensure fine_tuning.py has been executed successfully.")
        st.error(str(e))
        return None

def main():
    st.title("TestGenie")
    st.write("Enter a requirement below to generate the corresponding test case:")

    user_input = st.text_area("Requirement Input", "As a user, i want to log into the system using my credentials.")

    if st.button("Generate Test Case"):
        text_gen_pipeline = load_model_pipeline()
        if text_gen_pipeline is None:
            return

        # Wrap the pipeline with LangChain's LLM interface.
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

        # Define a prompt template.
        prompt_template = PromptTemplate(
            input_variables=["requirement"],
            template="Requirement: {requirement}\nTest Case:"
        )

        # Create a LangChain chain combining the LLM and the prompt.
        chain = LLMChain(llm=llm, prompt=prompt_template)

        try:
            output = chain.run(user_input)
            st.subheader("Generated Test Case")
            st.write(output)
        except Exception as e:
            st.error("Error generating test case:")
            st.error(str(e))

if __name__ == "__main__":
    main()
