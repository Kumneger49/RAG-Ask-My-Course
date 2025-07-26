import os
import logging  # For logging
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer  # For loading Mistral
import torch  # For tensor operations
from backend.vectorstore.retrieve import retrieve_top_k  # Import the retrieval function

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_answer.log'),
        logging.StreamHandler()
    ]
)

# Global variables for model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"  # Changed from mistralai/Mistral-7B-v0.1 to a more stable model
tokenizer = None
model = None

def load_models():
    """Load the Mistral model and tokenizer with proper authentication."""
    global tokenizer, model
    
    if tokenizer is not None and model is not None:
        return  # Already loaded
    
    logging.info(f"Loading DialoGPT model: {MODEL_NAME}")
    
    # Load HuggingFace token from environment
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if not hf_token:
        logging.warning("HUGGINGFACE_HUB_TOKEN not set, loading model without authentication")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    else:
        logging.info("Using HuggingFace token for authentication")
        # Pass token to model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=hf_token)
    
    logging.info("DialoGPT model loaded successfully.")


def build_prompt(chunks: List[Dict], question: str) -> str:
    """
    Builds a conversational prompt for DialoGPT using the retrieved context and the user's question.
    """
    # Only use the top 1 chunk for small-context models like DialoGPT
    context = chunks[0]['text'] if chunks else ""
    # For DialoGPT, treat context as the "previous message" and question as the "user message"
    prompt = f"{context}\nUser: {question}\nBot:"
    return prompt


def generate_answer(chunks: List[Dict], question: str, max_new_tokens: int = 100) -> str:
    load_models()
    prompt = build_prompt(chunks, question)
    logging.info("Prompt built for DialoGPT.")

    max_length = 1024
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the bot's reply (after "Bot:")
    if "Bot:" in full_response:
        answer = full_response.split("Bot:")[-1].strip()
    else:
        answer = full_response.strip()
    logging.info("DialoGPT answer generated.")
    return answer

# Example usage (uncomment to test):
if __name__ == "__main__":
    question = "What is a linear combination?"
    top_chunks = retrieve_top_k(question, k=5)
    answer = generate_answer(top_chunks, question)
    print("Answer:\n", answer) 