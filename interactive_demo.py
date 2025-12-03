"""
Interactive Demo for Telugu-English Code-Mixed Language Models
Test all models (N-gram, HMM, Hybrid, LSTM) with your own input
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ngram.ngram_model import UltraAdvancedNgramModel
from models.hmm.hmm_model import HMMModel
from models.hybrid.hybrid_language_tagger import HybridLanguageTagger
from models.lstm.lstm_model import LSTMLanguageModel


def load_models():
    """Load all trained models"""
    models = {}
    
    print("Loading models...")
    print("-" * 70)
    
    # Load N-gram model
    try:
        ngram_model = UltraAdvancedNgramModel(n=10)
        ngram_model.load('models/ngram/ngram_model.txt')
        models['ngram'] = ngram_model
        print("✓ N-gram model loaded")
    except Exception as e:
        print(f"✗ N-gram model failed to load: {e}")
        models['ngram'] = None
    
    # Load HMM model
    try:
        hmm_model = HMMModel()
        hmm_model.load_model('models/hmm_model.txt')
        models['hmm'] = hmm_model
        print("✓ HMM model loaded")
    except Exception as e:
        print(f"✗ HMM model failed to load: {e}")
        models['hmm'] = None
    
    # Load Hybrid model
    try:
        hybrid_model = HybridLanguageTagger.load('models/hybrid_tagger.txt')
        models['hybrid'] = hybrid_model
        print("✓ Hybrid model loaded")
    except Exception as e:
        print(f"✗ Hybrid model failed to load: {e}")
        models['hybrid'] = None
    
    # Load LSTM model
    try:
        lstm_model = LSTMLanguageModel()
        lstm_model.load_model('models/lstm/lstm_model.pkl')
        models['lstm'] = lstm_model
        print("✓ LSTM model loaded")
    except Exception as e:
        print(f"✗ LSTM model failed to load: {e}")
        models['lstm'] = None
    
    print("-" * 70)
    return models


def predict_with_all_models(models, context, top_k=5):
    """Get predictions from all models"""
    results = {}
    
    for model_name, model in models.items():
        if model is None:
            results[model_name] = []
            continue
        
        try:
            predictions = model.predict_next_word(context, top_k=top_k)
            results[model_name] = predictions
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            results[model_name] = []
    
    return results


def display_predictions(context, results, top_k=5):
    """Display predictions in a formatted way"""
    print("\n" + "=" * 70)
    print(f"Context: {' '.join(context)}")
    print("=" * 70)
    
    model_names = {
        'ngram': 'N-gram Model (10-gram)',
        'hmm': 'HMM Model (4-state)',
        'hybrid': 'Hybrid Model (N-gram + HMM)',
        'lstm': 'LSTM Model (2-layer, 2 epochs)'
    }
    
    for model_key, model_name in model_names.items():
        print(f"\n{model_name}:")
        print("-" * 70)
        
        if model_key not in results or not results[model_key]:
            print("  Model not available or no predictions")
            continue
        
        predictions = results[model_key][:top_k]
        
        if not predictions:
            print("  No predictions available")
            continue
        
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"  {i}. {word:20s} (probability: {prob:.6f})")
    
    print("\n" + "=" * 70)


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("  Telugu-English Code-Mixed Language Model - Interactive Demo")
    print("=" * 70)
    print("\n  This demo allows you to test all trained models with your own input.")
    print("  Enter Telugu-English code-mixed text and see predictions from:")
    print("    • N-gram Model (10-gram with Kneser-Ney smoothing)")
    print("    • HMM Model (4-state with Laplace smoothing)")
    print("    • Hybrid Model (Combined N-gram + HMM)")
    print("    • LSTM Model (2-layer pure Python implementation)")
    print("\n" + "=" * 70)


def print_examples():
    """Print example inputs"""
    print("\nExample inputs you can try:")
    print("  • nenu eppudu")
    print("  • how are you")
    print("  • chala bagundi")
    print("  • i will come")
    print("  • nenu velli untanu")
    print("  • what is your name")


def main():
    """Main interactive loop"""
    print_banner()
    
    # Load all models
    models = load_models()
    
    # Check if at least one model loaded successfully
    if all(model is None for model in models.values()):
        print("\n✗ Error: No models could be loaded!")
        print("Please ensure you have trained the models first.")
        print("Run: python train_all_models.py")
        return
    
    print_examples()
    
    print("\n" + "=" * 70)
    print("Ready! Type 'quit' or 'exit' to end the demo.")
    print("=" * 70)
    
    while True:
        try:
            # Get user input
            print("\n")
            user_input = input("Enter context words (space-separated): ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the interactive demo!")
                print("=" * 70)
                break
            
            # Skip empty input
            if not user_input:
                print("Please enter some context words.")
                continue
            
            # Parse context
            context = user_input.lower().split()
            
            # Validate context
            if len(context) < 1:
                print("Please enter at least one word.")
                continue
            
            # Get predictions
            results = predict_with_all_models(models, context, top_k=5)
            
            # Display results
            display_predictions(context, results, top_k=5)
            
            # Ask if user wants to continue
            print("\n" + "-" * 70)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            print("=" * 70)
            break
        
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again with different input.")


if __name__ == "__main__":
    main()
