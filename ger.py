import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from threading import Thread
import torch
import json
import os

# ============================================
# CONFIGURATION & THEME
# ============================================
CHAT_FILE = "chat_history.json"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ============================================
# DARK THEME CSS
# ============================================
DARK_THEME_CSS = """
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Chat container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Message containers */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* User message bubble */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 12px 16px !important;
        margin-left: auto !important;
        max-width: 80% !important;
        box-shadow: 0 2px 8px rgba(31, 111, 235, 0.3);
    }
    
    /* Assistant message bubble */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 12px 16px !important;
        max-width: 80% !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Chat input box */
    .stChatInputContainer {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 24px !important;
        padding: 4px !important;
    }
    
    .stChatInputContainer textarea {
        background-color: transparent !important;
        color: #e6edf3 !important;
        border: none !important;
    }
    
    .stChatInputContainer textarea::placeholder {
        color: #8b949e !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #58a6ff !important;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.2);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e6edf3 !important;
        font-family: 'Segoe UI', system-ui, sans-serif !important;
    }
    
    /* Text */
    p, span, div {
        color: #c9d1d9 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #161b22 !important;
        color: #a5d6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }
    
    pre {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    
    /* Typing indicator animation */
    @keyframes pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
    }
    
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #58a6ff;
        border-radius: 50%;
        margin: 0 2px;
        animation: pulse 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
</style>
"""

# ============================================
# STORAGE HELPERS
# ============================================
def save_chat(messages):
    with open(CHAT_FILE, "w") as file:
        json.dump(messages, file, indent=4)

def load_chat():
    if os.path.exists(CHAT_FILE) and os.path.getsize(CHAT_FILE) > 0:
        try:
            with open(CHAT_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            pass
    return [
        {
            "role": "system",
            "content": "You are a helpful and friendly AI assistant."
        }
    ]

# ============================================
# MODEL LOADING (Clean & Direct for CPU)
# ============================================
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("🔄 Loading model directly into RAM..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # No 4-bit dequantization overhead. Straightforward float32 matrix math for CPU execution.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
    return tokenizer, model

# ============================================
# GENERATION FUNCTION
# ============================================
def generate_response(messages, max_tokens, temperature):
    tokenizer, model = load_model()
    
    # Optional: Limiting history context size to prevent context slowdowns
    # Keeps only the system prompt and the last 5 messages
    context_messages = [messages[0]] + messages[-5:] if len(messages) > 6 else messages

    text = tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": True if temperature > 0.0 else False,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    return streamer

# ============================================
# SIDEBAR
# ============================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #e6edf3; margin-bottom: 4px;">🤖 Configuration</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### ⚙️ Generation Settings")
        
        # Reduced token thresholds to optimize generation speed on CPU architectures
        max_tokens = st.slider(
            "Max New Tokens",
            min_value=32,
            max_value=256,
            value=128,
            step=16
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.1
        )
        
        st.divider()
        st.markdown("### 🛠️ Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = [
                    {"role": "system", "content": "You are a helpful and friendly AI assistant."}
                ]
                save_chat(st.session_state.messages)
                st.rerun()
        with col2:
            if st.button("💾 Save", use_container_width=True):
                save_chat(st.session_state.messages)
                st.toast("Chat history saved!", icon="✅")
        
        return max_tokens, temperature

# ============================================
# MAIN APP
# ============================================
def main():
    st.set_page_config(
        page_title="Chikadibia AI Assistant Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat()
    
    max_tokens, temperature = render_sidebar()
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 8px;">
                💬 Chikadibia AI Assistant Chatbot
            </h1>
            <p style="color: #8b949e; font-size: 1rem;">
                Optimized Direct-RAM CPU Inference
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your message here...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat(st.session_state.messages)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            message_placeholder.markdown("""
                <div style="display: flex; align-items: center; gap: 8px; padding: 12px 0;">
                    <span style="color: #8b949e;">Thinking</span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                streamer = generate_response(
                    st.session_state.messages,
                    max_tokens,
                    temperature
                )
                
                for new_text in streamer:
                    full_response += new_text
                    if new_text.strip() or len(full_response) % 3 == 0:
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_chat(st.session_state.messages)
                
            except Exception as e:
                message_placeholder.error(f"❌ Generation error: {str(e)}")

if __name__ == "__main__":
    main()