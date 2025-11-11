"""
ULTRA-ADVANCED N-gram Model with Sequence Caching
Achieves MAXIMUM accuracy through exact sequence matching
"""

import os
import json
from collections import defaultdict, Counter
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils import config
from src.preprocessing.tokenizer import tokenize

class UltraAdvancedNgramModel:
    """
    ULTRA model with sequence caching for maximum accuracy
    
    New features:
    1. Stores complete sequences (up to 10 words)
    2. Exact sequence matching first
    3. Progressive backoff from 10-grams down to unigrams
    4. Smart caching for speed
    """
    
    def __init__(self, n=10, cache_sequences=True):
        self.n = n  # Up to 10-grams!
        self.counts_by_order = {k: {} for k in range(1, n+1)}
        self.vocabulary = set()
        
        # NEW: Sequence cache for exact matches
        self.sequence_cache = {}  # Maps context -> list of (next_word, count)
        self.cache_sequences = cache_sequences
        
    def train(self, sentences):
        print(f"Training ULTRA-ADVANCED {self.n}-gram model with sequence caching...")
        
        for sent_idx, sentence in enumerate(sentences):
            if sent_idx % 10000 == 0 and sent_idx > 0:
                print(f"  Processed {sent_idx} sentences...")
                
            tokens = sentence.split()
            self.vocabulary.update(tokens)

            L = len(tokens)
            
            # Count k-grams for all orders 1..n
            for i in range(L):
                for k in range(1, min(self.n + 1, L - i + 1)):
                    ngram = tuple(tokens[i:i + k])
                    self.counts_by_order[k][ngram] = self.counts_by_order[k].get(ngram, 0) + 1
                    
                    # Build sequence cache (for context lengths 1-9)
                    if self.cache_sequences and k >= 2 and k <= 10:
                        context = ngram[:-1]
                        next_word = ngram[-1]
                        if context not in self.sequence_cache:
                            self.sequence_cache[context] = {}
                        self.sequence_cache[context][next_word] = \
                            self.sequence_cache[context].get(next_word, 0) + 1
        
        print(f"✓ Vocabulary size: {len(self.vocabulary)}")
        print(f"✓ Sequence cache entries: {len(self.sequence_cache)}")
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if k <= self.n:
                count = len(self.counts_by_order.get(k, {}))
                if count > 0:
                    print(f"✓ Unique {k}-grams: {count}")
    
    def predict_next_word(self, context, top_k=5):
        """
        ULTRA prediction with sequence cache lookup
        """
        if isinstance(context, str):
            context = tokenize(context)
        
        context = list(context)
        
        # Use up to n-1 words as context
        max_context_len = self.n - 1
        if len(context) > max_context_len:
            context = context[-max_context_len:]
        
        def is_valid_token(tok):
            if not tok or not isinstance(tok, str):
                return False
            if not any(c.isalpha() for c in tok):
                return False
            if len(tok) < config.MIN_WORD_LENGTH or len(tok) > config.MAX_WORD_LENGTH:
                return False
            return True

        # ULTRA STRATEGY: Check sequence cache first!
        # Try from longest context down
        for ctx_len in range(min(len(context), 9), 0, -1):
            ctx_tuple = tuple(context[-ctx_len:])
            
            if ctx_tuple in self.sequence_cache:
                candidates = self.sequence_cache[ctx_tuple]
                
                # Sort by count
                sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                
                # Filter and score
                results = []
                total_count = sum(c for _, c in sorted_candidates)
                
                for word, count in sorted_candidates:
                    if is_valid_token(word):
                        # Apply exponential boost based on context length
                        # Longer context = much higher confidence
                        boost = (ctx_len ** 3) * count
                        results.append((word, boost))
                
                if results:
                    # Normalize to probabilities
                    total_score = sum(s for _, s in results)
                    results = [(w, s/total_score) for w, s in results]
                    return results[:top_k]
        
        # Fallback to top unigrams
        unigram_counts = self.counts_by_order.get(1, {})
        sorted_unigrams = sorted(unigram_counts.items(), key=lambda x: x[1], reverse=True)[:100]
        candidates = []
        for key, cnt in sorted_unigrams:
            w = key[0] if isinstance(key, tuple) else key
            if is_valid_token(w):
                candidates.append((w, cnt))
            if len(candidates) >= top_k:
                break
        
        total_count = sum(c for _, c in candidates) if candidates else 1
        return [(w, c/total_count) for w, c in candidates] if candidates else [('(no prediction)', 0.0)]
    
    def save(self, filepath):
        """Save model (including sequence cache)"""
        print(f"Saving ULTRA model to {filepath}...")
        
        vocab_dir = os.path.join(os.path.dirname(filepath), '..', 'data')
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, 'vocabulary.txt')

        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in sorted(self.vocabulary):
                f.write(f"{word}\n")

        # Save n-grams (only up to 5-grams to save space)
        counts_to_save = {}
        for k in range(1, min(6, self.n+1)):
            counts_to_save[str(k)] = {'|'.join(ng): cnt for ng, cnt in self.counts_by_order.get(k, {}).items()}
        
        # Save sequence cache
        cache_to_save = {
            '|'.join(ctx): {w: c for w, c in words.items()}
            for ctx, words in self.sequence_cache.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            data = {
                'counts_by_order': counts_to_save,
                'sequence_cache': cache_to_save,
                'vocabulary': list(self.vocabulary),
                'n': self.n
            }
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("✓ Model saved!")
    
    def load(self, filepath):
        """Load model from disk"""
        print(f"Loading ULTRA model from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.n = data.get('n', self.n)
            
            if 'counts_by_order' in data:
                self.counts_by_order = {
                    int(k): {tuple(ng.split('|')): cnt for ng, cnt in d.items()}
                    for k, d in data['counts_by_order'].items()
                }
            
            if 'sequence_cache' in data:
                self.sequence_cache = {
                    tuple(ctx.split('|')): {w: c for w, c in words.items()}
                    for ctx, words in data['sequence_cache'].items()
                }
            
            if 'vocabulary' in data:
                self.vocabulary = set(data['vocabulary'])
        
        print(f"✓ Loaded! Cache has {len(self.sequence_cache)} entries")


def train_ngram_model():
    """Train and save ULTRA N-gram model"""
    print("Loading training data...")
    with open("data/processed/train.processed.txt", 'r', encoding='utf-8') as f:
        train_sentences = [line.strip() for line in f if line.strip()][1:]  # Skip header
    
    print(f"Training on {len(train_sentences)} sentences...")
    model = UltraAdvancedNgramModel(n=10, cache_sequences=True)
    model.train(train_sentences)
    
    # Save model
    os.makedirs("models/ngram", exist_ok=True)
    model.save("models/ngram/ngram_model.txt")
    print("✓ Training complete!")
    
    return model


def load_ngram_model():
    """Load pre-trained N-gram model"""
    model = UltraAdvancedNgramModel(n=10)
    model.load("models/ngram/ngram_model.txt")
    return model


if __name__ == "__main__":
    train_ngram_model()
