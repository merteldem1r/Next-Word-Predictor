import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle
import os
from model import NextWordLSTM
from utils.data_utils import load_text_data, preprocess_text, create_vocabulary, create_sequences
from utils.metrics import calculate_accuracy, calculate_perplexity
from utils.visualization import plot_metrics
from config import (
    SEQUENCE_LENGTH, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, DATA_FILE, MODEL_PATH,
    VOCAB_PATH, RESULTS_DIR, TRAIN_VAL_SPLIT
)


# ==================== Training ====================
def train_model():
    """Main training function"""
    print("=" * 60)
    print("Next Word Predictor - Training")
    print("=" * 60)

    # Load and preprocess data
    print("\n1. Loading data...")
    text = load_text_data(DATA_FILE)
    words = preprocess_text(text)
    print(f"   Total words: {len(words)}")

    # Create vocabulary
    print("\n2. Creating vocabulary...")
    word_to_idx, idx_to_word = create_vocabulary(words)
    vocab_size = len(word_to_idx)
    print(f"   Vocabulary size: {vocab_size}")

    # Create sequences
    print("\n3. Creating training sequences...")
    sequences = create_sequences(words, word_to_idx, SEQUENCE_LENGTH)
    print(f"   Training samples: {len(sequences)}")

    if len(sequences) == 0:
        print("Not enough data to create sequences.")
        return

    # Prepare data for training
    X = torch.tensor([seq[0] for seq in sequences], dtype=torch.long)
    y = torch.tensor([seq[1] for seq in sequences], dtype=torch.long)
    dataset = TensorDataset(X, y)

    # Split into train and validation
    train_size = int(TRAIN_VAL_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {val_size}")

    # Create model
    print("\n4. Creating model...")
    model = NextWordLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    print(
        f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Select device (use Apple MPS when available for M1/M2 Macs)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"   Device: {device.type}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop with metrics tracking
    print("\n5. Training model...")
    print("-" * 60)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc_top1': [],
        'val_acc_top1': [],
        'train_acc_top5': [],
        'val_acc_top5': [],
        'train_perplexity': [],
        'val_perplexity': []
    }

    model.train()
    for epoch in range(EPOCHS):
        # Training phase
        total_loss = 0.0
        num_batches = 0
        print(f"Epoch [{epoch+1}/{EPOCHS}] starting...")

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # periodic progress update
            if (batch_idx + 1) % 500 == 0:
                print(
                    f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = total_loss / max(1, num_batches)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                val_batches += 1
        val_loss = val_loss / max(1, val_batches)
        model.train()

        # Calculate accuracies
        train_acc_top1, train_acc_top5 = calculate_accuracy(
            model, train_loader, device, top_k=5)
        val_acc_top1, val_acc_top5 = calculate_accuracy(
            model, val_loader, device, top_k=5)

        # Calculate perplexity
        train_perplexity = calculate_perplexity(train_loss)
        val_perplexity = calculate_perplexity(val_loss)

        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc_top1'].append(train_acc_top1)
        history['val_acc_top1'].append(val_acc_top1)
        history['train_acc_top5'].append(train_acc_top5)
        history['val_acc_top5'].append(val_acc_top5)
        history['train_perplexity'].append(train_perplexity)
        history['val_perplexity'].append(val_perplexity)

        print(f"Epoch [{epoch+1}/{EPOCHS}] completed")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Train Acc: {train_acc_top1*100:.2f}% | Val Acc: {val_acc_top1*100:.2f}%")
        print(
            f"  Perplexity: {train_perplexity:.2f} | Val Perplexity: {val_perplexity:.2f}")

    print("-" * 60)

    # Save model and vocabulary
    print("\n6. Generating visualizations...")
    from config import PLOT_FIGSIZE, PLOT_DPI
    plot_metrics(history, RESULTS_DIR, figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)

    print("\n7. Saving model and vocabulary...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    vocab_data = {
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': vocab_size,
        'seq_length': SEQUENCE_LENGTH
    }
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved: {VOCAB_PATH}")

    # Test prediction
    print("\n8. Testing prediction...")
    model.eval()
    test_sequence = words[:SEQUENCE_LENGTH]
    print(f"Input: {' '.join(test_sequence)}")

    test_indices = [word_to_idx.get(word, word_to_idx['<UNK>'])
                    for word in test_sequence]
    test_tensor = torch.tensor([test_indices], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(test_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_word = idx_to_word[predicted_idx]
        print(f"Predicted next word: {predicted_word}")

    print("\n" + "=" * 60)
    print("Training completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
