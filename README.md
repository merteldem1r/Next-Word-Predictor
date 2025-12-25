# Next Word Predictor

A Deep Learning project using LSTM neural networks to predict the next word in a sequence. Built for a Deep Learning class, featuring a complete pipeline from data preprocessing to a user-friendly web interface.

---

## Project Overview

This project implements a **character/word-level LSTM model** that learns from Shakespeare's complete works and predicts the next word based on input text. The model is trained using PyTorch with Apple MPS acceleration (optimized for M1/M2 Macs) and served through an interactive Gradio web interface.

**Key Features:**

- LSTM neural network with 2 layers (256 hidden units)
- Word embedding layer (128 dimensions)
- Shakespeare dataset (~202K words, 13.7K unique words)
- Train/Validation split with comprehensive metrics
- Real-time predictions with top-5 suggestions
- Training visualizations (loss, accuracy, perplexity curves)
- Fully modularized code structure
- Apple MPS GPU acceleration support

---

## What We Built

### The Complete Pipeline:

```
Raw Text Data → Preprocessing → Vocabulary → Sequences → LSTM Model → Predictions → Web UI
```

1. **Data Processing**: Load Shakespeare text, tokenize, create vocabulary
2. **Model Training**: Train LSTM on word sequences with validation metrics
3. **Evaluation**: Track loss, accuracy, and perplexity during training
4. **Visualization**: Generate training curves showing model performance
5. **Inference**: Load trained model and serve predictions via Gradio interface
6. **UI**: Interactive web app where users type words and see next word suggestions

---

## Project Structure

```
Next-Word-Predictor/
├── config/                      # Configuration management
│   ├── __init__.py
│   └── config.py               # All hyperparameters and paths
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── data_utils.py           # Data loading, preprocessing, vocabulary
│   ├── metrics.py              # Accuracy and evaluation functions
│   └── visualization.py        # Training visualization/plotting
│
├── data/                        # Data directory
│   └── shakespeare.txt          # Shakespeare corpus (~202K words)
│
├── saved_models/               # Trained model storage
│   ├── next_word_model.pth     # Model weights
│   └── vocabulary.pkl          # Word-to-index mappings
│
├── results/                    # Training outputs
│   ├── loss_curve.png          # Train/validation loss
│   ├── accuracy_top1_curve.png # Top-1 accuracy plot
│   ├── accuracy_top5_curve.png # Top-5 accuracy plot
│   └── perplexity_curve.png    # Perplexity curves
│
├── model.py                    # LSTM model architecture
├── train.py                    # Training script
├── serve.py                    # Gradio web interface
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## Getting Started

### Installation

1. **Create virtual environment:**

```bash
cd Next-Word-Predictor
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Dependencies:

- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computing
- `gradio>=4.0.0` - Web UI framework
- `matplotlib>=3.7.0` - Visualization

### Quick Start

**Train the model:**

```bash
python train.py
```

**Launch the web interface:**

```bash
python serve.py
```

Then open the URL shown (typically `http://127.0.0.1:7860`)

---

## Configuration

All hyperparameters are centralized in `config/config.py`:

**Model Architecture:**

```python
SEQUENCE_LENGTH = 5         # Input: 5 words → Output: 1 word
EMBEDDING_DIM = 128         # Word vector size
HIDDEN_DIM = 256            # LSTM hidden layer size
NUM_LAYERS = 2              # Stacked LSTM layers
DROPOUT = 0.3               # Regularization
```

**Training:**

```python
BATCH_SIZE = 128            # Samples per batch
EPOCHS = 10                 # Training iterations
LEARNING_RATE = 0.001       # Adam optimizer learning rate
TRAIN_VAL_SPLIT = 0.8       # 80% train, 20% validation
```

**Override via environment variables:**

```bash
export EPOCHS=20
export BATCH_SIZE=64
python train.py
```

---

## Training Results

After 10 epochs of training on Shakespeare:

| Metric             | Training     | Validation    |
| ------------------ | ------------ | ------------- |
| **Loss**           | 4.71         | 7.03          |
| **Top-1 Accuracy** | 19.03%       | 8.86%         |
| **Top-5 Accuracy** | 35.42%       | 18.74%        |
| **Perplexity**     | 110.61       | 1127.92       |
| **Vocab Size**     | 13,695 words | -             |
| **Training Time**  | ~5 minutes   | (M2 with MPS) |

**Example Prediction:**

```
Input:  "first citizen before we"
Output: "to" (predicted next word)
```

### Generated Visualizations

The training process generates 4 plots showing model learning:

1. **Loss Curve** - Shows train vs validation loss decreasing over epochs
2. **Accuracy (Top-1)** - Exact word prediction accuracy
3. **Accuracy (Top-5)** - Accuracy if correct word is in top 5 suggestions
4. **Perplexity** - Language model quality metric (lower is better)

See `results/` folder after training.

---

## How It Works

### 1. Data Processing (`utils/data_utils.py`)

- Loads raw text file
- Converts to lowercase and tokenizes into words
- Removes punctuation and special characters
- Creates vocabulary (word ↔ index mapping)
- Generates training sequences: `[word1, word2, word3, word4, word5] → word6`

### 2. Model Architecture (`model.py`)

```
Input Indices (5 words)
    ↓
Embedding Layer (128 dims)
    ↓
LSTM Layer 1 (256 hidden units)
    ↓
LSTM Layer 2 (256 hidden units)
    ↓
Dropout (0.3)
    ↓
Fully Connected (13,695 vocab size)
    ↓
Output: Probability distribution over vocabulary
```

### 3. Training (`train.py`)

- Splits data: 80% training, 20% validation
- Trains LSTM using cross-entropy loss
- Optimizes with Adam optimizer
- Tracks metrics each epoch:
  - Training & validation loss
  - Top-1 and Top-5 accuracy
  - Perplexity (model uncertainty)
- Saves best model to `saved_models/`
- Generates visualization plots

### 4. Inference & UI (`serve.py`)

- Loads trained model and vocabulary
- Takes user input text
- Preprocesses input (last 5 words)
- Gets prediction probabilities
- Returns top-5 most likely next words with confidence scores
- Serves via Gradio web interface

---

## Using the Web Interface

### Launch:

```bash
python serve.py
```

### Features:

- **Text Input**: Type any words (minimum 1, recommended 3-5)
- **Live Predictions**: Get top-5 next word suggestions
- **Confidence Scores**: Each suggestion shows probability percentage
- **Example Buttons**: Click to auto-fill example phrases
- **Share Link**: Gradio provides a public shareable URL

### Example Inputs:

```
"machine learning is"          → Suggests: "a", "the", "of", ...
"to be or not"                 → Suggests: "to", "be", "a", ...
"the future of technology is"  → Suggests: "very", "bright", ...
```

---

## Experimental Results & Analysis

### Experiment 1: Initial Training (20 Epochs, Batch Size 128)

**Settings:**
- EPOCHS = 20
- BATCH_SIZE = 128
- LEARNING_RATE = 0.001

**Key Observations:**

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Train Perplexity | Val Perplexity |
|-------|-----------|----------|-----------|---------|------------------|----------------|
| 1     | 6.64      | 6.34     | 8.69%     | 8.25%   | 765.57          | 566.61         |
| 5     | 5.46      | 6.55     | 12.87%    | 9.44%   | 234.05          | 700.98         |
| 8     | 5.02      | 7.00     | 15.48%    | 9.04%   | 150.83          | 1095.23        |

**What Went Wrong:**
- ⚠️ **Overfitting detected after epoch 5-6**
  - Training loss kept decreasing: 6.64 → 5.02
  - Validation loss started INCREASING: 6.34 → 7.00
  - Gap between train/val metrics grew wider
  
- ⚠️ **Validation perplexity exploded**
  - Started at 567, rose to 1095 by epoch 8
  - Model became increasingly confused on unseen data

- ⚠️ **Validation accuracy plateaued**
  - Barely improved from 8.25% to 9.04%
  - Training accuracy doubled (8.69% → 15.48%)

**Root Causes:**
1. **Too many epochs** - Model started memorizing training data
2. **Large batch size (128)** - Less frequent weight updates, worse generalization
3. **Shakespeare's archaic language** - Specialized vocabulary hard to generalize
4. **Model capacity** - 6.2M parameters might be too much for this dataset

**Lesson Learned:**
> "More training ≠ better model. Validation metrics must guide when to stop."

---

### Experiment 2: Optimized Training (In Progress)

**Adjusted Settings:**
- EPOCHS = 10 (reduced from 20)
- BATCH_SIZE = 64 (reduced from 128)
- LEARNING_RATE = 0.001 (unchanged)

**Expected Improvements:**
- Better generalization with smaller batches
- Stop before severe overfitting occurs
- Validation metrics should stabilize better

*Results will be updated after training completes...*

---

## Model Performance Analysis

### Strengths

- Model successfully learned word patterns from Shakespeare
- Training loss decreased consistently 
- Accuracy improved throughout training
- Model converged well (no catastrophic loss spikes)
- Fast training on M2 with MPS acceleration

### Limitations

- **Overfitting**: Large gap between train and validation accuracy
  - Model memorized training patterns too well
  - Shakespeare text is archaic/specialized

- **Low validation accuracy**: ~9% is modest
  - Large vocabulary (13.7K words) makes prediction harder
  - Model needs longer context for better predictions

- **High validation perplexity**: Indicates uncertainty on new data

### Why These Results?

1. **Small, specialized dataset** - Shakespeare is archaic language
2. **Short sequence length** - 5 words may not capture sufficient context
3. **Large vocabulary** - 13.7K unique words spread predictions thin
4. **Generic LSTM** - Transformer models would perform better

---

## Advanced Usage

### Train with different settings:

```bash
# Train longer with smaller batches
export EPOCHS=20
export BATCH_SIZE=64
python train.py
```

### Use different dataset:

1. Replace `data/shakespeare.txt` with your text file
2. Ensure it's valid UTF-8 text
3. Run `python train.py` (will create new vocabulary)

### Modify model architecture:

Edit `config/config.py`:

```python
EMBEDDING_DIM = 256        # Larger embeddings
HIDDEN_DIM = 512           # Larger LSTM
NUM_LAYERS = 3             # More layers (deeper)
```

---

## Dependencies

See `requirements.txt`:

```
torch>=2.0.0
numpy>=1.24.0
gradio>=4.0.0
matplotlib>=3.7.0
```
