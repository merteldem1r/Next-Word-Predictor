"""
Data utilities for text preprocessing and dataset preparation
"""
import os


def load_text_data(file_path):
    """Load text data from file"""
    if not os.path.exists(file_path):
        # Create sample data if file doesn't exist
        print(f"File not found. Creating sample data at {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        sample_text = """
        Machine learning is a subset of artificial intelligence. 
        Deep learning is a subset of machine learning.
        Neural networks are inspired by the human brain.
        Natural language processing helps computers understand text.
        Computer vision enables machines to interpret images.
        Artificial intelligence is transforming the world.
        Data science combines statistics and programming.
        Python is a popular programming language for data science.
        Deep learning models require large amounts of data.
        The future of technology is exciting and full of possibilities.
        """

        with open(file_path, 'w') as f:
            f.write(sample_text)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


def preprocess_text(text):
    """Clean and tokenize text"""
    # Convert to lowercase and split into words
    text = text.lower()
    # Remove extra whitespace and split
    words = text.split()
    # Basic cleaning
    words = [word.strip('.,!?;:"()[]') for word in words if word.strip()]
    return words


def create_vocabulary(words):
    """Create word to index and index to word mappings"""
    unique_words = sorted(set(words))

    # Add special tokens
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    idx_to_word = {0: '<PAD>', 1: '<UNK>'}

    # Add all unique words
    for i, word in enumerate(unique_words, start=2):
        word_to_idx[word] = i
        idx_to_word[i] = word

    return word_to_idx, idx_to_word


def create_sequences(words, word_to_idx, seq_length):
    """Create input-output sequences for training"""
    sequences = []

    for i in range(len(words) - seq_length):
        # Input: sequence of words
        input_seq = words[i:i + seq_length]
        # Output: next word
        target = words[i + seq_length]

        # Convert to indices
        input_indices = [word_to_idx.get(
            word, word_to_idx['<UNK>']) for word in input_seq]
        target_idx = word_to_idx.get(target, word_to_idx['<UNK>'])

        sequences.append((input_indices, target_idx))

    return sequences
