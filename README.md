# Non-Causal Transformer Chatbot

This repository contains a Streamlit-based chatbot application powered by a Non-Causal Transformer model. The chatbot is capable of generating (vaguely) human-like responses in real-time. The project implements a transformer-based encoder-decoder architecture with relative positional encodings.

## Features
- Non-causal transformer-based chatbot model.
- Fine-tuned for conversational tasks with frequency and repetition penalties.
- Deployed on Hugging Face Spaces for easy accessibility.
- Streamlit application for interactive user interaction.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/BaldeepArora/Non_Causal_Transformer_Chatbot.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Non_Causal_Transformer_Chatbot
    ```

3. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\\Scripts\\activate
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application Locally
1. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your browser and navigate to `http://localhost:8501`.

### Interacting with the Chatbot
- Type a message into the input box.
- The chatbot generates a response using the transformer model.
- Supports multi-turn conversations.

## Model Training
The Non-Causal Transformer was trained with the following parameters:
- Hidden Size: 512
- Number of Heads: 8
- Number of Layers: 6
- Feed-Forward Size: 2048
- Maximum Sequence Length: 128
- Number of Classes: (vocabulary size)
- Dropout: 0.1

### Training Steps
- Preprocessed conversational data was tokenized using a BERT tokenizer.
- GloVe embeddings were used to embed words and were trained later on.
- Relative positional encodings were added to the transformer model.
- The model was fine-tuned using a custom training loop with `AdamW` optimizer and learning rate scheduler.

## Deployment
The application is deployed on Hugging Face Spaces for public access.
- [Application Link](https://huggingface.co/spaces/baldeeparora/chatbot_non_causal_transformer)

### Steps to Deploy on Hugging Face Spaces
1. Push the repository to Hugging Face using `git-lfs` for large files.
2. Ensure the `requirements.txt` includes all dependencies.
3. Follow the [Hugging Face Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks) for deployment instructions.

## Dependencies
- Python 3.10+
- Streamlit
- PyTorch
- Transformers
- Hugging Face Hub
- scikit-learn
- Matplotlib
- NLTK
- Pandas
- NumPy

## Contributors
- **Baldeep Arora**



## Acknowledgments
- Hugging Face for the `transformers` library.
- PyTorch for the deep learning framework.
- Streamlit for the interactive app framework.
