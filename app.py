import streamlit as st
import torch
from transformers import BertTokenizer
from model import NonCausalTransformer  # Import your custom model class

# Parameters (Ensure these match your notebook)
hidden_size = 512
num_heads = 8
num_layers = 6
ff_size = 2048
num_classes = None  # This will be initialized after loading the tokenizer
dropout = 0.1
max_seq_len = 128

# Function to load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_path, tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    num_classes = tokenizer.vocab_size  # Update num_classes to match the tokenizer

    model = NonCausalTransformer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_size=ff_size,
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    model.load_state_dict(torch.load(f"{model_path}/Tuning4.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, tokenizer

# Load model and tokenizer
model_path = "non_causal_transformer"
tokenizer_path = "tokenizer"
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

# Response generation function
def generate_response_with_penalty(
    model,
    tokenizer,
    input_text,
    max_length=50,
    repetition_penalty=1.5,
    frequency_penalty=0.8
):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    input_ids = inputs["input_ids"]  # Shape: [batch_size, seq_len]
    attention_mask = inputs["attention_mask"]  # Shape: [batch_size, seq_len]

    response_ids = []  # This will store the generated tokens
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(input_ids, attention_mask, input_ids)
            logits = logits[:, -1, :]  # Focus on the last token

            # Apply repetition penalty
            if response_ids:
                for token_id in set(response_ids):
                    logits[0, token_id] /= repetition_penalty

            # Apply frequency penalty
            if response_ids:
                frequency_counts = torch.bincount(
                    torch.tensor(response_ids, dtype=torch.long), minlength=logits.shape[-1]
                )
                logits[0] -= frequency_penalty * frequency_counts

            # Get the next token
            next_token = torch.argmax(logits, dim=-1).item()

            if next_token == tokenizer.eos_token_id:
                break

            # Append the token and update input_ids and attention_mask
            response_ids.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.tensor([[1]])], dim=1
            )  # Update mask for the new token

    # Decode the generated response
    return tokenizer.decode(response_ids, skip_special_tokens=True)



# Streamlit app
st.title("Non-Causal Transformer Chatbot")
st.write("Have a conversation with the chatbot! It remembers the context.")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_area("Enter your message:", value="", placeholder="Type your message here...")

# Generate response
if st.button("Send"):
    if user_input.strip():
        # Append user input to conversation history
        st.session_state.history.append(f"User: {user_input}")

        # Prepare input for the model (combine history)
        conversation_context = " ".join(st.session_state.history)
        response = generate_response_with_penalty(
            model=model,
            tokenizer=tokenizer,
            input_text=conversation_context,
            max_length=50,
            repetition_penalty=1.5,
            frequency_penalty=0.8
        )

        # Append chatbot response to conversation history
        st.session_state.history.append(f"Chatbot: {response}")
    else:
        st.warning("Please enter a message.")

# Display conversation history
for message in st.session_state.history:
    st.write(message)
