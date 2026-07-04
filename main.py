from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from threading import Thread
import torch
import json
import os

CHAT_FILE = "chat_history.json"

# ---------------------------------
# Storage Helpers
# ---------------------------------
def save_chat(messages):
    with open(CHAT_FILE, "w") as file:
        json.dump(messages, file, indent=4)

def load_chat():
    # MODIFIED: Ensure file exists AND is not empty (size > 0)
    if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
        try:
            with open(CHAT_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            # Fallback if the file contains corrupted text
            pass
            
    # Fix 3: Proper default structure if no history file exists yet
    return [
        {
            "role": "system",
            "content": "You are a helpful and friendly AI assistant."
        }
    ]
# ---------------------------------
# Load the Model and Tokenizer
# ---------------------------------
def load_mind():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("✅ Model loaded successfully!\n")
    return tokenizer, model

# ---------------------------------
# Initialization Flow
# ---------------------------------
# Fix 1 & 2: Load history once cleanly without overwriting it later
messages = load_chat()
tokenizer, model = load_mind()

print("Chat initialized. Type 'exit' to quit.\n")

# ---------------------------------
# Chat Loop
# ---------------------------------
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("\nBOT: Goodbye! See you later.\n")
        break
    if not user_input.strip():
        continue
    # Save user message
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )
    # Convert conversation into text formatting required by the model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Tokenize inputs
    inputs = tokenizer(text, return_tensors="pt")
    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    # Generation settings
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 150,
        "temperature": 0.7,
        "do_sample": True
    }
    # Run generation in background thread
    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    thread.start()
    print("\nBOT: ", end="", flush=True)
    response = ""
    # Stream output token by token to the terminal
    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text
    print("\n")
    # Save assistant response to memory array
    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )    
    # Save updated conversation array directly to the JSON file
    save_chat(messages)