"""
Hybrid Language Model - HMM as Tagger + N-gram as Predictor

Architecture:
1. HMM tags the language of each word (Telugu/English/Mixed)
2. N-gram generates candidate predictions
3. Language tags boost/filter n-gram candidates

This is simpler and more effective than weighted interpolation.
"""

import json
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.ngram.ngram_model import UltraAdvancedNgramModel
from models.hmm.hmm_model import HMMModel


class HybridLanguageTagger:
    """
    Hybrid model using HMM for language tagging and N-gram for prediction.
    
    Key idea: Let each model do what it's best at:
    - HMM: Identify language (TE/EN) of words
    - N-gram: Generate predictions
    - Combination: Use language context to improve predictions
    """
    
    def __init__(self, n: int = 10):
        """
        Initialize hybrid model.
        
        Args:
            n: N-gram order (default 10)
        """
        self.n = n
        self.ngram_model = UltraAdvancedNgramModel(n=n)
        self.hmm_model = HMMModel()
        
        # Configuration
        self.boost_factor = 2.0  # Boost candidates matching predicted language
        self.min_confidence = 0.3  # Minimum HMM confidence to apply boost
        
        # Vocabulary (shared across models)
        self.vocab = set()
        
    def train(self, sentences: List[List[str]]):
        """
        Train both models.
        
        Args:
            sentences: List of tokenized sentences
        """
        # Both models expect strings, not lists
        sentences_str = [' '.join(sent) for sent in sentences]
        
        print("\n📚 Training N-gram model...")
        self.ngram_model.train(sentences_str)
        
        print("\n🏷️  Training HMM tagger...")
        self.hmm_model.train(sentences_str)
        
        # Build vocabulary
        self.vocab = self.ngram_model.vocabulary
        
        print(f"\n✅ Training complete!")
        print(f"   Vocabulary: {len(self.vocab):,} words")
        print(f"   N-gram order: {self.n}")
        print(f"   HMM states: {len(self.hmm_model.states)}")
        
    def _get_language_context(self, context: List[str]) -> Tuple[str, float]:
        """
        Use HMM to predict the most likely language for next word.
        
        Args:
            context: Previous words
            
        Returns:
            (predicted_language, confidence)
        """
        if not context:
            return "UNKNOWN", 0.0
            
        # Get HMM state sequence for context
        try:
            # HMM expects a list of words
            state_sequence = self.hmm_model._viterbi_decode(context)
            
            if not state_sequence or len(state_sequence) == 0:
                return "UNKNOWN", 0.0
                
            # Get last state (most relevant for next word)
            last_state = state_sequence[-1]
            
            # Extract language from state (e.g., "TE_COMMON" -> "TE")
            if last_state.startswith("TE"):
                lang = "TE"
            elif last_state.startswith("EN"):
                lang = "EN"
            else:
                lang = "UNKNOWN"
                
            # Simple confidence based on sequence consistency
            # If last 2-3 states are same language, high confidence
            confidence = 0.7  # Default moderate confidence
            if len(state_sequence) >= 2:
                last_langs = [s.split('_')[0] for s in state_sequence[-3:]]
                if all(l == lang for l in last_langs):
                    confidence = 0.9
                elif len(last_langs) >= 2 and last_langs[-2:] == [lang, lang]:
                    confidence = 0.8
            
            return lang, confidence
            
        except Exception as e:
            # Debug: print the error to understand what's wrong
            import sys
            print(f"DEBUG: HMM error for context {context}: {e}", file=sys.stderr)
            return "UNKNOWN", 0.0
    
    def _classify_word_language(self, word: str) -> str:
        """
        Classify a word as Telugu or English using HMM's trained language knowledge.
        
        For romanized text, character-based detection doesn't work.
        Use HMM's emission probabilities to determine language.
        
        Args:
            word: Word to classify
            
        Returns:
            "TE", "EN", "MIXED", or "UNKNOWN"
        """
        if not word:
            return "UNKNOWN"
        
        # Check if has non-ASCII (native script)
        has_telugu_chars = any(ord(c) > 127 for c in word)
        if has_telugu_chars:
            return "TE"
        
        # If not in HMM vocabulary, use n-gram knowledge
        if word not in self.hmm_model.vocabulary:
            # Check if word appears more in Telugu or English contexts in n-gram
            # Look at the word's overall frequency in different language patterns
            
            # Simple heuristic: check common English words
            common_english = {'the', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
                            'do', 'does', 'did', 'will', 'would', 'could', 'should',
                            'can', 'may', 'might', 'must', 'i', 'you', 'he', 'she',
                            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                            'my', 'your', 'his', 'her', 'its', 'our', 'their',
                            'this', 'that', 'these', 'those', 'am', 'be', 'been',
                            'being', 'a', 'an', 'and', 'or', 'but', 'if', 'then',
                            'so', 'as', 'at', 'by', 'for', 'from', 'in', 'of',
                            'on', 'to', 'with', 'about', 'hello', 'hi', 'how', 'what',
                            'when', 'where', 'who', 'why', 'ok', 'okay', 'yes', 'no'}
            
            if word.lower() in common_english:
                return "EN"
            
            # Otherwise assume Telugu for romanized text
            return "TE"
        
        # Use HMM's emission probabilities to classify
        te_common_prob = self.hmm_model.emit_prob.get('TE_COMMON', {}).get(word, 0)
        te_rare_prob = self.hmm_model.emit_prob.get('TE_RARE', {}).get(word, 0)
        en_common_prob = self.hmm_model.emit_prob.get('EN_COMMON', {}).get(word, 0)
        en_rare_prob = self.hmm_model.emit_prob.get('EN_RARE', {}).get(word, 0)
        
        te_total = te_common_prob + te_rare_prob
        en_total = en_common_prob + en_rare_prob
        
        # Determine language based on which has higher emission probability
        if te_total > en_total * 1.5:  # Telugu clearly dominant
            return "TE"
        elif en_total > te_total * 1.5:  # English clearly dominant
            return "EN"
        elif te_total > 0 and en_total > 0:  # Both present
            return "MIXED"
        elif te_total > 0:
            return "TE"
        elif en_total > 0:
            return "EN"
        else:
            return "UNKNOWN"
    
    def predict_next_word(
        self,
        context: List[str],
        top_k: int = 5,
        use_language_boost: bool = True
    ) -> List[Tuple[str, float]]:
        """
        ADVANCED HYBRID: Research-based approach (Chandu et al., 2018)
        
        Key insight from "Language Informed Modeling of Code-Switched Text":
        Encoding language information at CODE-SWITCHING POINTS is crucial!
        
        Strategy (Factored Language Model + Re-ranking):
        1. Get N-gram predictions (strong contextual model)
        2. Detect if we're at a code-switch boundary
        3. If at code-switch: boost HMM's language transition knowledge
        4. If not: trust N-gram more
        5. Re-rank based on language coherence
        
        Args:
            context: Previous words
            top_k: Number of predictions to return
            use_language_boost: Whether to use hybrid approach
            
        Returns:
            List of (word, score) tuples
        """
        if not context:
            return self.ngram_model.predict_next_word(context, top_k=top_k)
        
        # Get N-gram predictions (primary model)
        ngram_preds = self.ngram_model.predict_next_word(context, top_k=top_k * 10)
        
        if not use_language_boost or not ngram_preds:
            return ngram_preds[:top_k]
        
        # Get HMM predictions for language patterns
        hmm_preds = self.hmm_model.predict_next_word(context, top_k=top_k * 10)
        
        # STEP 1: Detect code-switching context
        # Tag recent words' languages
        recent_context = context[-3:] if len(context) >= 3 else context
        context_langs = []
        for word in recent_context:
            lang = self._classify_word_language(word)
            if lang != "UNKNOWN":
                context_langs.append(lang)
        
        # Detect if we're at potential code-switch boundary
        is_code_switch_boundary = False
        if len(context_langs) >= 2:
            # Check if languages are mixed in recent context
            unique_langs = set(context_langs)
            if len(unique_langs) > 1:
                is_code_switch_boundary = True
            # Or if last two words are different languages
            elif len(context_langs) >= 2 and context_langs[-1] != context_langs[-2]:
                is_code_switch_boundary = True
        
        # STEP 2: Adaptive interpolation based on context type
        context_len = len(context)
        
        if is_code_switch_boundary:
            # At code-switch: HMM's language model helps MORE
            # Research shows language info crucial at switch points
            if context_len >= 5:
                lambda_weight = 0.75  # More HMM influence
            elif context_len >= 3:
                lambda_weight = 0.70
            else:
                lambda_weight = 0.60  # Short context + code-switch: use both equally
        else:
            # Not at code-switch: N-gram is very strong
            if context_len >= 7:
                lambda_weight = 0.95  # Trust N-gram heavily
            elif context_len >= 5:
                lambda_weight = 0.92
            elif context_len >= 3:
                lambda_weight = 0.88
            else:
                lambda_weight = 0.80
        
        # STEP 3: Build candidate pool
        all_candidates = {}
        
        for word, score in ngram_preds:
            all_candidates[word] = {'ngram': score, 'hmm': 0.0}
        
        for word, score in hmm_preds:
            if word in all_candidates:
                all_candidates[word]['hmm'] = score
            else:
                all_candidates[word] = {'ngram': 0.0, 'hmm': score}
        
        # STEP 4: Interpolate + Re-rank with language coherence
        scored_candidates = []
        
        for word, scores in all_candidates.items():
            ngram_score = scores['ngram']
            hmm_score = scores['hmm']
            
            # Base interpolated score
            hybrid_score = (lambda_weight * ngram_score) + ((1 - lambda_weight) * hmm_score)
            
            # Re-ranking factor: Language coherence bonus
            candidate_lang = self._classify_word_language(word)
            
            if context_langs:
                dominant_lang = max(set(context_langs), key=context_langs.count)
                
                if is_code_switch_boundary:
                    # At code-switch: slightly prefer switching to other language
                    # (code-mixing is natural, not random)
                    if candidate_lang != "UNKNOWN" and candidate_lang != dominant_lang:
                        hybrid_score *= 1.05  # Small boost for natural switch
                else:
                    # Not at switch: prefer continuing same language
                    if candidate_lang == dominant_lang:
                        hybrid_score *= 1.10  # Boost for language continuity
            
            # Consensus bonus: both models agree
            if ngram_score > 0.6 and hmm_score > 0.4:
                hybrid_score *= 1.15  # Strong consensus
            elif ngram_score > 0.4 and hmm_score > 0.6:
                hybrid_score *= 1.10  # HMM-led consensus
            
            scored_candidates.append((word, hybrid_score))
        
        # Sort and normalize
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if scored_candidates and scored_candidates[0][1] > 0:
            max_score = scored_candidates[0][1]
            scored_candidates = [(word, score / max_score) for word, score in scored_candidates]
        
        return scored_candidates[:top_k]
    
    def save(self, filepath: str):
        """
        Save hybrid model to file.
        
        Args:
            filepath: Path to save model
        """
        print(f"\n💾 Saving hybrid model to {filepath}...")
        
        # Save both models separately
        base_path = filepath.rsplit('.', 1)[0]
        
        # Save N-gram
        ngram_path = f"{base_path}_ngram.txt"
        self.ngram_model.save(ngram_path)
        print(f"   ✓ N-gram saved to {ngram_path}")
        
        # Save HMM
        hmm_path = f"{base_path}_hmm.txt"
        self.hmm_model.save_model(hmm_path)
        print(f"   ✓ HMM saved to {hmm_path}")
        
        # Save metadata
        metadata = {
            'n': self.n,
            'boost_factor': self.boost_factor,
            'min_confidence': self.min_confidence,
            'vocab_size': len(self.vocab),
            'ngram_path': ngram_path,
            'hmm_path': hmm_path
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✓ Metadata saved to {filepath}")
        print(f"\n✅ Hybrid model saved successfully!")
    
    @staticmethod
    def load(filepath: str) -> 'HybridLanguageTagger':
        """
        Load hybrid model from file.
        
        Args:
            filepath: Path to model metadata file
            
        Returns:
            Loaded HybridLanguageTagger
        """
        print(f"\n📂 Loading hybrid model from {filepath}...")
        
        # Load metadata
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Create model
        model = HybridLanguageTagger(n=metadata['n'])
        model.boost_factor = metadata['boost_factor']
        model.min_confidence = metadata['min_confidence']
        
        # Load N-gram
        print(f"   Loading N-gram from {metadata['ngram_path']}...")
        model.ngram_model = UltraAdvancedNgramModel(n=metadata['n'])
        model.ngram_model.load(metadata['ngram_path'])
        
        # Load HMM
        print(f"   Loading HMM from {metadata['hmm_path']}...")
        model.hmm_model = HMMModel()
        model.hmm_model.load_model(metadata['hmm_path'])
        
        # Restore vocabulary
        model.vocab = model.ngram_model.vocabulary
        
        print(f"   ✓ Vocabulary: {len(model.vocab):,} words")
        print(f"   ✓ N-gram order: {model.n}")
        print(f"   ✓ HMM states: {len(model.hmm_model.states)}")
        print(f"\n✅ Hybrid model loaded successfully!")
        
        return model


def main():
    """Test hybrid model."""
    print("🧪 Testing Hybrid Language Tagger Model\n")
    
    # Simple test data
    test_sentences = [
        ['hello', 'how', 'are', 'you'],
        ['nenu', 'bagunanu', 'thank', 'you'],
        ['ela', 'unnav', 'bro'],
        ['i', 'am', 'fine', 'thanks'],
    ]
    
    print("Training on test data...")
    model = HybridLanguageTagger(n=3)
    model.train(test_sentences)
    
    # Test predictions
    print("\n🔮 Testing predictions:")
    test_contexts = [
        ['hello', 'how'],
        ['nenu', 'bagunanu'],
        ['ela'],
    ]
    
    for context in test_contexts:
        print(f"\nContext: {context}")
        
        # Get language prediction
        lang, conf = model._get_language_context(context)
        print(f"Predicted language: {lang} (confidence: {conf:.2f})")
        
        # Get predictions
        preds = model.predict_next_word(context, top_k=3)
        print("Predictions:")
        for word, score in preds:
            word_lang = model._classify_word_language(word)
            print(f"  {word:15s} {score:.4f} [{word_lang}]")
    
    print("\n✅ Test complete!")


if __name__ == '__main__':
    main()
