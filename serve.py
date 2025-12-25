import torch
import pickle
import os
import gradio as gr
from model import NextWordLSTM
from config import (
    MODEL_PATH, VOCAB_PATH, EMBEDDING_DIM, HIDDEN_DIM,
    NUM_LAYERS, DROPOUT, TOP_K_PREDICTIONS
)


# ==================== Load Model and Vocabulary ====================
def load_model_and_vocab():
    """Load trained model and vocabulary"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please train the model first by running train.py")

    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(
            f"Vocabulary not found at {VOCAB_PATH}. Please train the model first by running train.py")

    # Load vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        vocab_data = pickle.load(f)

    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = vocab_data['idx_to_word']
    vocab_size = vocab_data['vocab_size']
    seq_length = vocab_data['seq_length']

    # Load model
    model = NextWordLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    print("Model and vocabulary loaded successfully.")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {seq_length}")

    return model, word_to_idx, idx_to_word, seq_length


# ==================== Prediction Functions ====================
def predict_next_words(model, input_text, word_to_idx, idx_to_word, seq_length, top_k=5):
    """
    Predict top K next words based on input text

    Args:
        model: Trained LSTM model
        input_text: Input text string
        word_to_idx: Word to index mapping
        idx_to_word: Index to word mapping
        seq_length: Required sequence length
        top_k: Number of predictions to return

    Returns:
        List of (word, probability) tuples
    """
    # Preprocess input
    words = input_text.lower().strip().split()

    if len(words) == 0:
        return []

    # Take last seq_length words
    if len(words) >= seq_length:
        input_words = words[-seq_length:]
    else:
        # Pad with special token if too short
        input_words = ['<PAD>'] * (seq_length - len(words)) + words

    # Convert to indices
    input_indices = [word_to_idx.get(
        word, word_to_idx.get('<UNK>', 1)) for word in input_words]
    input_tensor = torch.tensor([input_indices], dtype=torch.long)

    # Get predictions
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    # Get top K predictions
    top_probs, top_indices = torch.topk(
        probabilities, min(top_k, len(probabilities)))

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        word = idx_to_word.get(idx.item(), '<UNK>')
        # Skip special tokens
        if word not in ['<PAD>', '<UNK>']:
            predictions.append((word, prob.item() * 100))

    return predictions


# ==================== Gradio Interface ====================
def create_autocomplete_interface():
    """Create Gradio interface for next word prediction"""

    try:
        model, word_to_idx, idx_to_word, seq_length = load_model_and_vocab()
    except FileNotFoundError as e:
        print(str(e))
        print("\nPlease run 'python train.py' first to train the model.")
        return None

    def predict_interface(text):
        """Interface function for Gradio"""
        if not text or not text.strip():
            return "Start typing to see word suggestions."

        predictions = predict_next_words(
            model, text, word_to_idx, idx_to_word, seq_length, top_k=TOP_K_PREDICTIONS)

        if not predictions:
            return "No predictions available. Try typing more words."

        # Format output
        result = "Next Word Suggestions:\n\n"
        for i, (word, prob) in enumerate(predictions, 1):
            result += f"{i}. **{word}** - {prob:.1f}% confidence\n"

        return result

    # Create Gradio interface
    interface = gr.Interface(
        fn=predict_interface,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Start typing here... (e.g., 'machine learning is')",
            label="Input Text"
        ),
        outputs=gr.Markdown(label="Predictions"),
        title="Next Word Predictor",
        description="""
        Deep Learning Project - Text Prediction with LSTM
        
        Type a few words to get suggestions for the next word.
        The model uses an LSTM neural network trained on text data.
        
        Tip: Type at least 3-5 words for better predictions.
        """,
        examples=[
            ["machine learning is"],
            ["artificial intelligence"],
            ["deep learning"],
            ["natural language"],
            ["the future of"]
        ]
    )

    return interface


# ==================== Main ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Next Word Predictor Server")
    print("=" * 60)
    print()

    interface = create_autocomplete_interface()

    if interface:
        print("\nLaunching Gradio interface...")
        print("=" * 60)
        interface.launch(share=True)
    else:
        print("\nFailed to start server. Please train the model first:")
        print("   python train.py")
