import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.tokenizer import tokenize
from src.utils import config

class LanguageTagger:
    """
    ENHANCED Language Tagger with multi-feature analysis.
    Uses comprehensive patterns from actual dataset analysis.
    """
    def __init__(self):
        # Common Telugu words (from dataset frequency analysis)
        self.telugu_common = {
            'ka', 'na', 'oka', 'tana', 'aite', 'chala', 'ayana', 'una', 'vari', 
            'mi', 'telugu', 'ma', 'pa', 'ame', 'ga', 'ika', 'rendu', 'mandi',
            'ko', 'vishayam', 'miru', 'vundi', 'varu', 'chesina', 'nundi',
            'madya', 'vala', 'telangana', 'samayam', 'te', 'prabutvam', 'manchi',
            'ane', 'varaku', 'ade', 'naku', 'kata', 'kota', 'pedda', 'china',
            'chitram', 'prati', 'da', 'jila', 'kadu', 'meeru', 'memu', 'vaadu',
            'aayana', 'idi', 'adi', 'ela', 'ekkada', 'eppudu', 'nenu',
            'ki', 'lo', 'ni', 'nu', 'ku', 'tho', 'nunchi', 'enduku', 'ela',
            'chesi', 'chesu', 'undi', 'andi', 'unna', 'anna', 'ante', 'anta',
            'mana', 'mee', 'valla', 'vala', 'kuda', 'kani', 'kani', 'kani'
        }
        
        # English words (from dataset)
        self.english_common = {
            'party', 'cinema', 'corona', 'low', 'movie', 'weekend', 'office', 
            'gym', 'shopping', 'coffee', 'breakfast', 'lunch', 'dinner',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'the', 'a', 'an',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can',
            'this', 'that', 'these', 'those', 'what', 'where', 'when', 'why',
            'how', 'who', 'which', 'not', 'no', 'yes', 'ok', 'okay'
        }
        
        # Telugu endings (from dataset analysis - top patterns)
        self.telugu_endings = {
            'am', 'na', 'la', 'ga', 'ta', 'ti', 'ka', 'ku', 'di', 'du', 
            'ru', 'da', 'nu', 'te', 'ri', 'ra', 'ma', 'ya', 'tu', 'ni', 
            'va', 'hi', 'li', 'lo', 'ki', 'tho', 'ndi', 'vu', 'nna', 'nnu',
            'anu', 'alu', 'ala', 'ela', 'ila', 'ulu', 'ula'
        }
        
        # Telugu beginnings (from dataset analysis)
        self.telugu_beginnings = {
            'ch', 'ma', 'pa', 'sa', 'ka', 'pr', 'va', 'vi', 'ta', 'ba', 
            'an', 'ra', 'na', 'te', 'da', 'ni', 'ko', 'un', 'ja', 've', 
            'pe', 'ad', 'ga', 'po', 'mu', 'mi'
        }
        
        # Character features
        self.telugu_chars = set('aiueoktdmnrpvslgbhyjw')  # Common in Telugu
        self.english_indicators = set('qxz')  # Rare in Telugu
        
    def _calculate_score(self, word):
        """
        Calculate Telugu score for a word.
        Returns: positive = more Telugu, negative = more English
        """
        word_lower = word.lower()
        score = 0
        
        # 1. Check common word lists (HIGHEST WEIGHT)
        if word_lower in self.telugu_common:
            return 100  # Definitely Telugu
        if word_lower in self.english_common:
            return -100  # Definitely English
        
        # 2. Check endings (STRONG SIGNAL - 2-3 char endings)
        if len(word) >= 2:
            ending2 = word_lower[-2:]
            if ending2 in self.telugu_endings:
                score += 40
        
        if len(word) >= 3:
            ending3 = word_lower[-3:]
            # Check 3-char Telugu patterns
            if ending3 in ['amu', 'ana', 'ala', 'ulu', 'ula', 'ina', 'una']:
                score += 35
        
        # 3. Check beginnings
        if len(word) >= 2:
            begin2 = word_lower[:2]
            if begin2 in self.telugu_beginnings:
                score += 25
        
        # 4. Character analysis
        # Telugu words have high vowel-to-consonant ratio
        vowels = sum(1 for c in word_lower if c in 'aeiou')
        consonants = len(word_lower) - vowels
        if consonants > 0:
            vowel_ratio = vowels / len(word_lower)
            if vowel_ratio > 0.4:  # Telugu is vowel-heavy
                score += 20
        
        # 5. English indicators
        if any(c in self.english_indicators for c in word_lower):
            score -= 50  # Strong English signal
        
        # 6. Consecutive consonants (more common in English)
        consonant_clusters = 0
        for i in range(len(word_lower) - 1):
            if word_lower[i] not in 'aeiou' and word_lower[i+1] not in 'aeiou':
                consonant_clusters += 1
        if consonant_clusters > 2:
            score -= 15
        
        # 7. Double letters (common in Telugu: ll, nn, tt)
        if any(word_lower[i] == word_lower[i+1] for i in range(len(word_lower)-1)):
            double_letters = [word_lower[i] for i in range(len(word_lower)-1) 
                            if word_lower[i] == word_lower[i+1]]
            if any(dl in 'lnt' for dl in double_letters):
                score += 15  # Telugu pattern
        
        return score
    
    def tag_word(self, word):
        """Tag a word as Telugu (TE) or English (EN)"""
        score = self._calculate_score(word)
        return 'TE' if score >= 0 else 'EN'
    
    def tag_with_confidence(self, word):
        """Returns (tag, confidence_score)"""
        score = self._calculate_score(word)
        tag = 'TE' if score >= 0 else 'EN'
        confidence = abs(score) / 100.0  # Normalize to 0-1+
        return tag, min(confidence, 1.0)

class HMMModel:
    """
    REVOLUTIONARY HMM Model with research-backed improvements:
    1. Class-based emissions (common vs rare words)
    2. Bigram emissions P(word | prev_word, state)
    3. TF-IDF weighting for important words
    4. Viterbi decoding (best path vs forward sum)
    5. Multiple granularity states
    """
    def __init__(self):
        # Enhanced state space
        self.states = ['TE_COMMON', 'TE_RARE', 'EN_COMMON', 'EN_RARE']
        self.lang_states = {'TE_COMMON': 'TE', 'TE_RARE': 'TE', 
                           'EN_COMMON': 'EN', 'EN_RARE': 'EN'}
        
        self.start_prob = {}
        self.trans_prob = {}
        self.emit_prob = {}  # P(word | state)
        self.emit_bigram_prob = {}  # P(word | prev_word, state) - NEW!
        
        self.vocabulary = set()
        self.tagger = LanguageTagger()
        
        # Word frequency for classification
        self.word_freq = {}
        self.common_threshold = 10  # Words appearing 10+ times are "common"
        
        # TF-IDF scores for word importance
        self.word_importance = {}
        
        # Word transitions
        self.word_transitions = defaultdict(lambda: defaultdict(int))
        self.word_language_scores = {}
    
    def _classify_word_rarity(self, word):
        """Classify if word is common or rare"""
        freq = self.word_freq.get(word, 0)
        return 'COMMON' if freq >= self.common_threshold else 'RARE'
    
    def _get_state(self, word):
        """Get the appropriate state for a word"""
        lang = self.tagger.tag_word(word)
        rarity = self._classify_word_rarity(word)
        return f"{lang}_{rarity}"
    
    def train(self, sentences, num_sentences=None):
        """REVOLUTIONARY training with class-based emissions and bigrams"""
        print("Training REVOLUTIONARY HMM model...")
        
        if num_sentences is None:
            num_sentences = len(sentences)
        
        # PHASE 1: Count word frequencies (for common/rare classification)
        print("  Phase 1: Analyzing word frequencies...")
        for sentence in sentences[:num_sentences]:
            if not sentence or sentence == 'processed_text':
                continue
            tokens = sentence.split()
            for word in tokens:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
                self.vocabulary.add(word)
        
        # Calculate TF-IDF importance scores
        print("  Phase 2: Computing word importance (TF-IDF)...")
        import math
        total_docs = num_sentences
        doc_freq = {}  # How many sentences contain each word
        for sentence in sentences[:num_sentences]:
            if not sentence or sentence == 'processed_text':
                continue
            unique_words = set(sentence.split())
            for word in unique_words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
        
        # TF-IDF: important words appear frequently but not in every sentence
        for word in self.vocabulary:
            tf = self.word_freq[word] / sum(self.word_freq.values())
            idf = math.log(total_docs / (doc_freq.get(word, 1) + 1))
            self.word_importance[word] = tf * idf
        
        # PHASE 3: Count state transitions and emissions
        print("  Phase 3: Building state model...")
        state_counts = {s: 0 for s in self.states}
        trans_counts = {s1: {s2: 0 for s2 in self.states} for s1 in self.states}
        emit_counts = {s: defaultdict(int) for s in self.states}
        emit_bigram_counts = {s: defaultdict(lambda: defaultdict(int)) for s in self.states}
        
        word_lang_evidence = defaultdict(lambda: {'TE': 0, 'EN': 0})
        
        for sentence in sentences[:num_sentences]:
            if not sentence or sentence == 'processed_text':
                continue
            
            tokens = sentence.split()
            if len(tokens) == 0:
                continue
            
            # Get states for each word
            states_seq = [self._get_state(word) for word in tokens]
            
            # Track language evidence
            for word, state in zip(tokens, states_seq):
                tag, confidence = self.tagger.tag_with_confidence(word)
                word_lang_evidence[word][tag] += confidence
                word_lang_evidence[word]['TE' if tag == 'EN' else 'EN'] += (1 - confidence) * 0.2
            
            # Count initial states
            state_counts[states_seq[0]] += 1
            
            # Count transitions, emissions, and bigram emissions
            for i in range(len(tokens)):
                state = states_seq[i]
                word = tokens[i]
                
                # Emission: P(word | state)
                emit_counts[state][word] += 1
                
                if i > 0:
                    # Transition: P(state_i | state_i-1)
                    prev_state = states_seq[i-1]
                    trans_counts[prev_state][state] += 1
                    
                    # Bigram emission: P(word_i | word_i-1, state_i)
                    prev_word = tokens[i-1]
                    emit_bigram_counts[state][prev_word][word] += 1
                    
                    # Word transitions
                    self.word_transitions[prev_word][word] += 1
        
        # Store language evidence
        for word, evidence in word_lang_evidence.items():
            total = evidence['TE'] + evidence['EN']
            if total > 0:
                self.word_language_scores[word] = {
                    'TE': evidence['TE'] / total,
                    'EN': evidence['EN'] / total
                }
        
        # PHASE 4: Calculate probabilities with TF-IDF weighting
        print("  Phase 4: Computing probabilities...")
        
        # Start probabilities
        total_starts = sum(state_counts.values())
        if total_starts > 0:
            self.start_prob = {s: (count + 0.1) / (total_starts + 0.4) 
                             for s, count in state_counts.items()}
        else:
            # Default: prefer common Telugu
            self.start_prob = {
                'TE_COMMON': 0.6, 'TE_RARE': 0.3,
                'EN_COMMON': 0.08, 'EN_RARE': 0.02
            }
        
        # Transition probabilities with smoothing
        for s1 in self.states:
            total_trans = sum(trans_counts[s1].values())
            if total_trans > 0:
                self.trans_prob[s1] = {
                    s2: (count + 0.01) / (total_trans + 0.04)
                    for s2, count in trans_counts[s1].items()
                }
            else:
                # Default: prefer staying in same language, same rarity
                self.trans_prob[s1] = {s2: 0.0 for s2 in self.states}
                # Stay in same state: 0.7
                self.trans_prob[s1][s1] = 0.7
                # Same language, different rarity: 0.2
                same_lang = [s for s in self.states if self.lang_states[s] == self.lang_states[s1] and s != s1]
                for s in same_lang:
                    self.trans_prob[s1][s] = 0.2 / len(same_lang) if same_lang else 0
                # Different language: 0.1
                diff_lang = [s for s in self.states if self.lang_states[s] != self.lang_states[s1]]
                for s in diff_lang:
                    self.trans_prob[s1][s] = 0.1 / len(diff_lang) if diff_lang else 0
        
        # Emission probabilities with TF-IDF weighting
        for state in self.states:
            total_emit = sum(emit_counts[state].values())
            if total_emit > 0:
                self.emit_prob[state] = {}
                for word, count in emit_counts[state].items():
                    base_prob = count / total_emit
                    # Boost by TF-IDF importance
                    importance = self.word_importance.get(word, 0.5)
                    # Boost by language confidence
                    lang = self.lang_states[state]
                    lang_conf = self.word_language_scores.get(word, {}).get(lang, 0.5)
                    # Combined: corpus freq (60%) + importance (20%) + confidence (20%)
                    self.emit_prob[state][word] = (
                        0.6 * base_prob +
                        0.2 * (base_prob * (1 + importance)) +
                        0.2 * (base_prob * (1 + lang_conf))
                    )
            else:
                self.emit_prob[state] = {}
        
        # Bigram emission probabilities (NEW!)
        for state in self.states:
            self.emit_bigram_prob[state] = {}
            for prev_word, next_words in emit_bigram_counts[state].items():
                total_following = sum(next_words.values())
                if total_following > 0:
                    self.emit_bigram_prob[state][prev_word] = {
                        word: count / total_following
                        for word, count in next_words.items()
                    }
        
        print(f"  ✓ Vocabulary: {len(self.vocabulary)}")
        print(f"  ✓ Common words: {sum(1 for f in self.word_freq.values() if f >= self.common_threshold)}")
        print(f"  ✓ States: {self.states}")
        print(f"  ✓ Start probs: {dict(sorted(self.start_prob.items(), key=lambda x: x[1], reverse=True))}")
        
        # Show emission counts per state
        for state in self.states:
            print(f"  ✓ {state} emissions: {len(self.emit_prob[state])}")
    
    def _viterbi_decode(self, context):
        """
        Viterbi algorithm: Find BEST state sequence (not sum of all paths).
        This gives much sharper, more confident predictions.
        """
        if not context:
            return self.start_prob.copy()
        
        # Initialize
        V = [{} for _ in range(len(context))]
        path = {}
        
        # Initial probabilities for first word
        for state in self.states:
            emit_p = self.emit_prob.get(state, {}).get(context[0], config.HMM_SMOOTHING_ALPHA)
            # Boost by language confidence
            if context[0] in self.word_language_scores:
                lang = self.lang_states[state]
                lang_conf = self.word_language_scores[context[0]].get(lang, 0.5)
                emit_p *= (1 + lang_conf * 0.5)
            V[0][state] = self.start_prob.get(state, 1e-10) * emit_p
            path[state] = [state]
        
        # Iterate through remaining observations
        for t in range(1, len(context)):
            new_path = {}
            
            for curr_state in self.states:
                # Find most likely previous state
                max_prob = 0
                best_prev_state = None
                
                for prev_state in self.states:
                    trans_p = self.trans_prob.get(prev_state, {}).get(curr_state, 1e-10)
                    prob = V[t-1][prev_state] * trans_p
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_state
                
                # Emission probability with bigram boost
                emit_p = self.emit_prob.get(curr_state, {}).get(context[t], config.HMM_SMOOTHING_ALPHA)
                
                # BIGRAM EMISSION BOOST (NEW!)
                prev_word = context[t-1]
                if (curr_state in self.emit_bigram_prob and 
                    prev_word in self.emit_bigram_prob[curr_state] and
                    context[t] in self.emit_bigram_prob[curr_state][prev_word]):
                    bigram_p = self.emit_bigram_prob[curr_state][prev_word][context[t]]
                    # Interpolate: 70% bigram, 30% unigram
                    emit_p = 0.7 * bigram_p + 0.3 * emit_p
                
                # Language confidence boost
                if context[t] in self.word_language_scores:
                    lang = self.lang_states[curr_state]
                    lang_conf = self.word_language_scores[context[t]].get(lang, 0.5)
                    emit_p *= (1 + lang_conf * 0.5)
                
                V[t][curr_state] = max_prob * emit_p
                new_path[curr_state] = path[best_prev_state] + [curr_state]
            
            path = new_path
        
        # Find best final state
        max_prob = max(V[-1].values())
        # Normalize to get posterior probabilities
        total = sum(V[-1].values()) or 1.0
        posterior = {state: prob/total for state, prob in V[-1].items()}
        
        return posterior
    
    def predict_next_word(self, context, top_k=5):
        """REVOLUTIONARY prediction using Viterbi + bigram emissions + TF-IDF"""
        if isinstance(context, str):
            context = tokenize(context)
        
        # Empty context handling
        if not context:
            candidates = set()
            # Focus on common states (they have more useful emissions)
            for s in ['TE_COMMON', 'EN_COMMON']:
                sorted_emits = sorted(self.emit_prob.get(s, {}).items(),
                                    key=lambda x: x[1], reverse=True)[:top_k*10]
                candidates.update(w for w, _ in sorted_emits)
            
            preds = []
            alpha = config.HMM_SMOOTHING_ALPHA
            for w in candidates:
                score = 0.0
                for s in self.states:
                    p_emit = self.emit_prob.get(s, {}).get(w, alpha)
                    p_start = self.start_prob.get(s, 0.0)
                    # Boost by TF-IDF importance
                    importance = self.word_importance.get(w, 0.5)
                    score += p_start * p_emit * (1 + importance)
                preds.append((w, score))
            preds.sort(key=lambda x: x[1], reverse=True)
            return preds[:top_k]

        # Use VITERBI for best state sequence (not forward sum!)
        posterior = self._viterbi_decode(context)

        # Build enhanced candidate set
        candidates = set()
        last_word = context[-1]
        
        # Strategy 1: BIGRAM EMISSIONS (highest priority!)
        for state in self.states:
            if posterior.get(state, 0) > 0.05:  # Only probable states
                if (state in self.emit_bigram_prob and 
                    last_word in self.emit_bigram_prob[state]):
                    bigram_words = sorted(
                        self.emit_bigram_prob[state][last_word].items(),
                        key=lambda x: x[1], reverse=True
                    )[:top_k*10]
                    candidates.update(w for w, _ in bigram_words)
        
        # Strategy 2: Word transitions (corpus-level bigrams)
        if last_word in self.word_transitions:
            top_trans = sorted(self.word_transitions[last_word].items(),
                             key=lambda x: x[1], reverse=True)[:top_k*8]
            candidates.update(w for w, _ in top_trans)
        
        # Strategy 3: High-probability emissions for probable states
        for state in self.states:
            if posterior.get(state, 0) > 0.1:
                sorted_emits = sorted(self.emit_prob.get(state, {}).items(),
                                    key=lambda x: x[1], reverse=True)[:top_k*12]
                candidates.update(w for w, _ in sorted_emits)
        
        # Strategy 4: Language-appropriate words
        # Determine dominant language from posterior
        lang_scores = {'TE': 0, 'EN': 0}
        for state, prob in posterior.items():
            lang_scores[self.lang_states[state]] += prob
        dominant_lang = 'TE' if lang_scores['TE'] >= lang_scores['EN'] else 'EN'
        
        # Add common words from dominant language
        for state in [f"{dominant_lang}_COMMON"]:
            sorted_emits = sorted(self.emit_prob.get(state, {}).items(),
                                key=lambda x: x[1], reverse=True)[:top_k*8]
            candidates.update(w for w, _ in sorted_emits)

        # Score all candidates with multi-factor model
        preds = []
        alpha_smooth = config.HMM_SMOOTHING_ALPHA
        
        for w in candidates:
            score = 0.0
            
            # Component 1: Bigram emission (HIGHEST WEIGHT - 50%)
            bigram_score = 0.0
            for state in self.states:
                if (state in self.emit_bigram_prob and 
                    last_word in self.emit_bigram_prob[state] and
                    w in self.emit_bigram_prob[state][last_word]):
                    bigram_score += posterior.get(state, 0) * self.emit_bigram_prob[state][last_word][w]
            
            # Component 2: Unigram emission (25%)
            unigram_score = 0.0
            for state in self.states:
                p_emit = self.emit_prob.get(state, {}).get(w, alpha_smooth)
                unigram_score += posterior.get(state, 0) * p_emit
            
            # Component 3: Word transition boost (15%)
            trans_score = 0.0
            if last_word in self.word_transitions and w in self.word_transitions[last_word]:
                trans_count = self.word_transitions[last_word][w]
                total_trans = sum(self.word_transitions[last_word].values())
                trans_score = trans_count / total_trans
            
            # Component 4: TF-IDF importance (10%)
            importance_score = self.word_importance.get(w, 0.5)
            
            # COMBINED SCORE
            score = (0.50 * bigram_score + 
                    0.25 * unigram_score + 
                    0.15 * trans_score +
                    0.10 * importance_score)
            
            # Additional boost for language match
            w_lang = self.tagger.tag_word(w)
            if w_lang == dominant_lang:
                score *= 1.2
            
            preds.append((w, score))

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:top_k] if preds else [('(no prediction)', 0.0)]
        
        # Strategy 1: Words that follow last context word (transition-based)
        last_word = context[-1]
        if last_word in self.word_transitions:
            top_transitions = sorted(self.word_transitions[last_word].items(),
                                   key=lambda x: x[1], reverse=True)[:top_k*5]
            candidates.update(w for w, _ in top_transitions)
        
        # Strategy 2: High-probability emissions for probable states
        for s in self.states:
            if posterior.get(s, 0) > 0.1:  # Only consider probable states
                sorted_emits = sorted(self.emit_prob.get(s, {}).items(), 
                                    key=lambda x: x[1], reverse=True)[:top_k*15]
                candidates.update(w for w, _ in sorted_emits)
        
        # Strategy 3: Language-appropriate words based on context language
        context_lang_scores = {'TE': 0, 'EN': 0}
        for word in context[-3:]:  # Look at last 3 words
            if word in self.word_language_scores:
                for lang in ['TE', 'EN']:
                    context_lang_scores[lang] += self.word_language_scores[word].get(lang, 0)
        
        # Determine dominant language
        dominant_lang = 'TE' if context_lang_scores['TE'] >= context_lang_scores['EN'] else 'EN'
        # Add words from dominant language
        sorted_emits = sorted(self.emit_prob.get(dominant_lang, {}).items(),
                            key=lambda x: x[1], reverse=True)[:top_k*10]
        candidates.update(w for w, _ in sorted_emits)

        # Score candidates using posterior + transitions + language confidence
        preds = []
        alpha_smooth = config.HMM_SMOOTHING_ALPHA
        
        for w in candidates:
            score = 0.0
            
            # Component 1: HMM emission probability weighted by posterior
            for s in self.states:
                p_emit = self.emit_prob.get(s, {}).get(w, alpha_smooth)
                p_state = posterior.get(s, 0.0)
                score += p_state * p_emit
            
            # Component 2: Word transition boost (if follows last word)
            if last_word in self.word_transitions and w in self.word_transitions[last_word]:
                trans_count = self.word_transitions[last_word][w]
                total_trans = sum(self.word_transitions[last_word].values())
                trans_boost = trans_count / total_trans
                score *= (1 + trans_boost * 2.0)  # Significant boost for observed transitions
            
            # Component 3: Language confidence boost
            if w in self.word_language_scores:
                # Boost words that match context language
                lang_conf = self.word_language_scores[w].get(dominant_lang, 0.5)
                score *= (1 + lang_conf * 0.5)
            
            preds.append((w, score))

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:top_k]

    def posterior_states(self, context):
        if isinstance(context, str):
            context = tokenize(context)
        if not context:
            return self.start_prob.copy()

        alpha_prev = {}
        first_obs = context[0]
        for s in self.states:
            emit_p = self.emit_prob.get(s, {}).get(first_obs, config.HMM_SMOOTHING_ALPHA)
            alpha_prev[s] = self.start_prob.get(s, 0.0) * emit_p
        total = sum(alpha_prev.values()) or 1.0
        for s in alpha_prev:
            alpha_prev[s] /= total

        for obs in context[1:]:
            alpha_curr = {s: 0.0 for s in self.states}
            for s_prev in self.states:
                for s_curr in self.states:
                    trans_p = self.trans_prob.get(s_prev, {}).get(s_curr, 0.0)
                    alpha_curr[s_curr] += alpha_prev[s_prev] * trans_p
            for s in self.states:
                emit_p = self.emit_prob.get(s, {}).get(obs, config.HMM_SMOOTHING_ALPHA)
                alpha_curr[s] *= emit_p
            total = sum(alpha_curr.values()) or 1.0
            for s in alpha_curr:
                alpha_curr[s] /= total
            alpha_prev = alpha_curr

        return alpha_prev

    def score_candidates(self, context, candidates, alpha_smooth=None):
        if alpha_smooth is None:
            alpha_smooth = config.HMM_SMOOTHING_ALPHA
        posterior = self.posterior_states(context)
        scores = {}
        for w in candidates:
            score = 0.0
            for s in self.states:
                p_emit = self.emit_prob.get(s, {}).get(w, alpha_smooth)
                score += posterior.get(s, 0.0) * p_emit
            scores[w] = score
        return scores
    
    def save_model(self, filepath):
        """Save revolutionary model with all new features"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            data = {
                'states': self.states,
                'lang_states': self.lang_states,
                'start_prob': self.start_prob,
                'trans_prob': self.trans_prob,
                'emit_prob': self.emit_prob,
                'emit_bigram_prob': {
                    s: {w1: dict(w2_dict) for w1, w2_dict in bigrams.items()}
                    for s, bigrams in self.emit_bigram_prob.items()
                },
                'vocabulary': list(self.vocabulary),
                'word_freq': self.word_freq,
                'word_importance': self.word_importance,
                'word_language_scores': self.word_language_scores,
                'word_transitions': {k: dict(v) for k, v in self.word_transitions.items()},
                'common_threshold': self.common_threshold
            }
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_model(self, filepath):
        """Load revolutionary model with all new features"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.states = data['states']
            self.lang_states = data.get('lang_states', {'TE_COMMON': 'TE', 'TE_RARE': 'TE', 
                                                        'EN_COMMON': 'EN', 'EN_RARE': 'EN'})
            self.start_prob = data['start_prob']
            self.trans_prob = data['trans_prob']
            self.emit_prob = data['emit_prob']
            
            # Load bigram emissions
            emit_bigram_data = data.get('emit_bigram_prob', {})
            self.emit_bigram_prob = {}
            for state, bigrams in emit_bigram_data.items():
                self.emit_bigram_prob[state] = {}
                for w1, w2_dict in bigrams.items():
                    self.emit_bigram_prob[state][w1] = w2_dict
            
            self.vocabulary = set(data['vocabulary'])
            self.word_freq = data.get('word_freq', {})
            self.word_importance = data.get('word_importance', {})
            self.word_language_scores = data.get('word_language_scores', {})
            
            # Load word transitions
            word_trans_data = data.get('word_transitions', {})
            self.word_transitions = defaultdict(lambda: defaultdict(int))
            for w1, transitions in word_trans_data.items():
                for w2, count in transitions.items():
                    self.word_transitions[w1][w2] = count
            
            self.common_threshold = data.get('common_threshold', 10)

def train_hmm_model():
    # Load training data
    print("Loading training data...")
    with open("data/processed/train.processed.txt", 'r', encoding='utf-8') as f:
        train_data = [line.strip() for line in f if line.strip()]
    
    # Train model
    model = HMMModel()
    model.train(train_data)
    
    # Save model
    print("\nSaving model...")
    model.save_model("models/hmm_model.txt")
    print("Done!")
    return model

def load_hmm_model():
    model = HMMModel()
    model.load_model("models/hmm_model.txt")
    return model

if __name__ == "__main__":
    train_hmm_model()