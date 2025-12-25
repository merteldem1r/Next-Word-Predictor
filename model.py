import torch
import torch.nn as nn


class NextWordLSTM(nn.Module):
    """
    LSTM model for next word prediction
    Architecture: Embedding -> LSTM -> Fully Connected -> Output
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size: Size of vocabulary (number of unique words)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(NextWordLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer: converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: processes sequential data
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer: maps LSTM output to vocabulary
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of word indices, shape (batch_size, sequence_length)
        Returns:
            Output tensor of shape (batch_size, vocab_size)
        """
        # Convert word indices to embeddings
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim)

        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Apply dropout
        last_output = self.dropout(last_output)

        # Map to vocabulary size
        output = self.fc(last_output)  # (batch_size, vocab_size)

        return output
