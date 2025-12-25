"""
Configuration file for Next Word Predictor
Contains all hyperparameters, paths, and settings
"""
import os


# ==================== Model Hyperparameters ====================
SEQUENCE_LENGTH = 5      # Number of words to use for prediction
EMBEDDING_DIM = 128      # Word embedding dimension
HIDDEN_DIM = 256         # LSTM hidden layer size
NUM_LAYERS = 2           # Number of LSTM layers
DROPOUT = 0.3            # Dropout rate for regularization


# ==================== Training Parameters ====================
# Allow override via environment variables for easy experimentation
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
LEARNING_RATE = 0.001


# ==================== Data Paths ====================
DATA_FILE = "data/shakespeare.txt"
MODEL_PATH = "saved_models/next_word_model.pth"
VOCAB_PATH = "saved_models/vocabulary.pkl"
RESULTS_DIR = "results"


# ==================== Evaluation Settings ====================
TOP_K_PREDICTIONS = 5    # Number of predictions to show in UI
TRAIN_VAL_SPLIT = 0.8    # Train/validation split ratio


# ==================== Visualization Settings ====================
PLOT_DPI = 150           # Resolution for saved plots
PLOT_FIGSIZE = (10, 6)   # Figure size for plots
