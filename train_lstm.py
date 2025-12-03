"""
Train LSTM Language Model from scratch
Pure Python implementation for Telugu-English code-mixed text
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lstm.lstm_model import LSTMLanguageModel


def load_training_data(filepath):
    """Load and tokenize training data"""
    print(f"Loading training data from {filepath}...")
    
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words = line.lower().split()
                if len(words) > 1:  # Skip single-word lines
                    texts.append(words)
    
    print(f"Loaded {len(texts)} sequences")
    return texts


def main():
    print("=" * 70)
    print("LSTM Language Model Training (Pure Python - No Frameworks)")
    print("=" * 70)
    
    # Paths
    train_file = 'data/processed/train.processed.txt'
    model_save_path = 'models/lstm/lstm_model.pkl'
    
    # Check if training file exists
    if not os.path.exists(train_file):
        print(f"Error: Training file not found: {train_file}")
        return
    
    # Load training data
    start_time = time.time()
    train_texts = load_training_data(train_file)
    
    if not train_texts:
        print("Error: No training data loaded!")
        return
    
    # Initialize LSTM model
    print("\nInitializing LSTM model...")
    model = LSTMLanguageModel(
        vocab_size=10000,      # Maximum vocabulary size
        embedding_dim=128,      # Word embedding dimension
        hidden_size=256,        # LSTM hidden state size
        num_layers=2,           # Number of LSTM layers
        learning_rate=0.001,    # Learning rate
        context_length=10       # Number of context words
    )
    
    # Build vocabulary from training data
    model.build_vocabulary(train_texts, min_freq=2)
    
    # Initialize weights
    model.initialize_weights()
    
    # Train model
    print("\nStarting training...")
    model.train(
        train_texts=train_texts,
        epochs=2,               # Number of training epochs
        batch_size=32,          # Batch size
        save_path=model_save_path,
        n_workers=1             # Sequential training for stability
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Model saved to: {model_save_path}")
    
    # Test predictions
    print("\n" + "=" * 70)
    print("Testing predictions on sample contexts:")
    print("=" * 70)
    
    test_contexts = [
        ["nenu", "eppudu", "ranu"],
        ["how", "are", "you"],
        ["nenu", "velli", "untanu"],
        ["i", "will", "come"],
        ["chala", "bagundi"]
    ]
    
    for context in test_contexts:
        predictions = model.predict_next_word(context, top_k=5)
        print(f"\nContext: {' '.join(context)}")
        print("Top 5 predictions:")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"  {i}. {word:20s} (prob: {prob:.6f})")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
