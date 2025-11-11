"""
Train All Models - N-gram, HMM, and Hybrid

Trains all three models on the same training data and saves them.
"""

import time
import sys


def train_ngram():
    """Train N-gram model"""
    print("\n" + "=" * 70)
    print("1️⃣  TRAINING N-GRAM MODEL (10-gram)")
    print("=" * 70)
    
    from models.ngram.ngram_model import train_ngram_model
    
    start = time.time()
    train_ngram_model()
    elapsed = time.time() - start
    
    print(f"\n✅ N-gram training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    

def train_hmm():
    """Train HMM model"""
    print("\n" + "=" * 70)
    print("2️⃣  TRAINING HMM MODEL (4-state)")
    print("=" * 70)
    
    from models.hmm.hmm_model import train_hmm_model
    
    start = time.time()
    train_hmm_model()
    elapsed = time.time() - start
    
    print(f"\n✅ HMM training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


def train_hybrid():
    """Train Hybrid model"""
    print("\n" + "=" * 70)
    print("3️⃣  TRAINING HYBRID MODEL (HMM Tagger + N-gram)")
    print("=" * 70)
    
    from models.hybrid.hybrid_language_tagger import HybridLanguageTagger
    
    # Load training data
    print("\n📖 Loading training data...")
    with open('data/processed/train.processed.txt', 'r', encoding='utf-8') as f:
        train_sentences = [line.strip().split() for line in f if line.strip()]
    
    print(f"   Loaded {len(train_sentences):,} sentences")
    
    # Train
    start = time.time()
    model = HybridLanguageTagger(n=10)
    model.train(train_sentences)
    
    # Save
    model.save('models/hybrid_tagger.txt')
    
    elapsed = time.time() - start
    print(f"\n✅ Hybrid training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")


def main():
    """Train all models"""
    print("=" * 70)
    print("🚀 TRAINING ALL MODELS")
    print("=" * 70)
    
    overall_start = time.time()
    
    # Train all three models
    try:
        train_ngram()
        train_hmm()
        train_hybrid()
        
        overall_time = time.time() - overall_start
        
        print("\n" + "=" * 70)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nTotal training time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
        print("\nModels saved:")
        print("  • models/ngram/ngram_model.txt")
        print("  • models/hmm_model.txt")
        print("  • models/hybrid_tagger.txt")
        print("\nNext step: Run evaluate_all_models.py to compare performance")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
