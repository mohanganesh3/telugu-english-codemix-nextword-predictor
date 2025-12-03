"""
Evaluate LSTM Language Model
Compare performance with N-gram, HMM, and Hybrid models
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lstm.lstm_model import LSTMLanguageModel


def load_test_data(filepath):
    """Load test data"""
    print(f"Loading test data from {filepath}...")
    
    test_sequences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words = line.lower().split()
                if len(words) > 1:
                    test_sequences.append(words)
    
    print(f"Loaded {len(test_sequences)} test sequences")
    return test_sequences


def evaluate_model(model, test_sequences, context_length=10):
    """
    Evaluate model on test sequences
    
    Returns:
        results: Dictionary with evaluation metrics
    """
    print("\nEvaluating LSTM model...")
    
    total_predictions = 0
    top1_correct = 0
    top5_correct = 0
    mrr_sum = 0.0
    
    start_time = time.time()
    
    for seq_idx, sequence in enumerate(test_sequences):
        # Make predictions for each position in sequence (start from position 1)
        for i in range(1, len(sequence)):
            # Use up to context_length words as context
            context = sequence[max(0, i - context_length):i]
            target = sequence[i]
            
            # Skip if no context
            if not context:
                continue
            
            # Get predictions
            predictions = model.predict_next_word(context, top_k=5)
            
            if not predictions:
                continue
            
            # Extract predicted words
            predicted_words = [word for word, _ in predictions]
            
            # Check if target in predictions
            if target in predicted_words:
                rank = predicted_words.index(target) + 1
                
                if rank == 1:
                    top1_correct += 1
                    top5_correct += 1
                    mrr_sum += 1.0
                elif rank <= 5:
                    top5_correct += 1
                    mrr_sum += 1.0 / rank
            
            total_predictions += 1
            
            # Progress update
            if total_predictions % 5000 == 0:
                current_top1 = (top1_correct / total_predictions) * 100
                print(f"  Progress: {total_predictions} predictions | Top-1: {current_top1:.2f}%")
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    top1_accuracy = (top1_correct / total_predictions) * 100 if total_predictions > 0 else 0
    top5_accuracy = (top5_correct / total_predictions) * 100 if total_predictions > 0 else 0
    mrr = mrr_sum / total_predictions if total_predictions > 0 else 0
    
    results = {
        'total_predictions': total_predictions,
        'top1_correct': top1_correct,
        'top5_correct': top5_correct,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'mrr': mrr,
        'time': elapsed_time
    }
    
    return results


def main():
    print("=" * 70)
    print("LSTM Language Model Evaluation")
    print("=" * 70)
    
    # Paths
    model_path = 'models/lstm/lstm_model.pkl'
    test_file = 'data/processed/test.processed.txt'
    results_file = 'models/lstm/lstm_results.txt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using train_lstm.py")
        return
    
    # Check if test file exists
    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        return
    
    # Load model
    print(f"\nLoading LSTM model from {model_path}...")
    model = LSTMLanguageModel()
    model.load_model(model_path)
    
    # Load test data
    test_sequences = load_test_data(test_file)
    
    if not test_sequences:
        print("Error: No test data loaded!")
        return
    
    # Evaluate model
    results = evaluate_model(model, test_sequences, context_length=model.context_length)
    
    # Print results
    print("\n" + "=" * 70)
    print("LSTM MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total predictions:  {results['total_predictions']:,}")
    print(f"Top-1 correct:      {results['top1_correct']:,}")
    print(f"Top-5 correct:      {results['top5_correct']:,}")
    print(f"Top-1 Accuracy:     {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy:     {results['top5_accuracy']:.2f}%")
    print(f"MRR:                {results['mrr']:.4f}")
    print(f"Evaluation time:    {results['time']:.2f}s")
    print("=" * 70)
    
    # Save results to file
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LSTM LANGUAGE MODEL EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: LSTM (2 layers, hidden_size={model.hidden_size})\n")
        f.write(f"Context length: {model.context_length}\n")
        f.write(f"Vocabulary size: {model.vocab_size}\n")
        f.write(f"Embedding dim: {model.embedding_dim}\n\n")
        f.write(f"Total predictions:  {results['total_predictions']:,}\n")
        f.write(f"Top-1 correct:      {results['top1_correct']:,}\n")
        f.write(f"Top-5 correct:      {results['top5_correct']:,}\n")
        f.write(f"Top-1 Accuracy:     {results['top1_accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy:     {results['top5_accuracy']:.2f}%\n")
        f.write(f"MRR:                {results['mrr']:.4f}\n")
        f.write(f"Evaluation time:    {results['time']:.2f}s\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Compare with previous best (Hybrid: 7.33%)
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS BEST MODEL (Hybrid)")
    print("=" * 70)
    print(f"Hybrid Top-1:       7.33%")
    print(f"LSTM Top-1:         {results['top1_accuracy']:.2f}%")
    
    improvement = results['top1_accuracy'] - 7.33
    if improvement > 0:
        print(f"Improvement:        +{improvement:.2f}% (LSTM is better!)")
    elif improvement < 0:
        print(f"Difference:         {improvement:.2f}% (Hybrid is still better)")
    else:
        print(f"Same performance")
    print("=" * 70)


if __name__ == "__main__":
    main()
