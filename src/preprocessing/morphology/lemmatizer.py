from typing import Dict, Set
from ..preprocess_config import *

class TeluguEnglishLemmatizer:
    def __init__(self):
        # English irregular mappings
        self.english_irregulars = {
            # Be forms
            'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be', 'been': 'be',
            # Have forms
            'has': 'have', 'have': 'have', 'had': 'have',
            # Do forms
            'does': 'do', 'did': 'do', 'done': 'do',
            # Say forms
            'says': 'say', 'said': 'say',
            # Make forms
            'makes': 'make', 'made': 'make',
            # Take forms
            'takes': 'take', 'took': 'take', 'taken': 'take',
            # See forms
            'sees': 'see', 'saw': 'see', 'seen': 'see',
            # Know forms
            'knows': 'know', 'knew': 'know', 'known': 'know',
            # Get forms
            'gets': 'get', 'got': 'get', 'gotten': 'get',
            # Give forms
            'gives': 'give', 'gave': 'give', 'given': 'give',
            # Tell forms
            'tells': 'tell', 'told': 'tell',
            # Find forms
            'finds': 'find', 'found': 'find',
            # Think forms
            'thinks': 'think', 'thought': 'think',
            # Come forms
            'comes': 'come', 'came': 'come',
            # Go forms
            'goes': 'go', 'went': 'go', 'gone': 'go',
            # Irregular nouns
            'children': 'child',
            'people': 'person',
            'mice': 'mouse',
            'geese': 'goose',
            'teeth': 'tooth',
            'feet': 'foot',
            'lives': 'life',
            'wives': 'wife',
            'leaves': 'leaf',
            'wolves': 'wolf',
            'shelves': 'shelf'
        }

        # Telugu patterns and endings from config
        self.telugu_patterns = TELUGU_PATTERNS
        self.telugu_verb_endings = TELUGU_VERB_ENDINGS
        self.telugu_noun_endings = TELUGU_NOUN_ENDINGS
        
    def is_telugu_word(self, word: str) -> bool:
        word = word.lower()
        # Prefer stronger Telugu signals to avoid false positives from short
        # consonant clusters that also appear in English (e.g. 'ch', 'th').
        # Look for long vowel patterns or geminated consonants first.
        long_vowels = ('aa', 'ee', 'oo')
        geminates = ('nnu', 'mmu', 'llu', 'ddu', 'nchi')

        if any(v in word for v in long_vowels + geminates):
            return True

        # Otherwise rely on well-known Telugu verb/noun endings.
        if any(word.endswith(ending) for ending in self.telugu_verb_endings):
            return True

        if any(word.endswith(ending) for ending in self.telugu_noun_endings):
            return True

        return False

    def lemmatize_english(self, word: str) -> str:
        word = word.lower()
        
        # Check irregular forms first
        if word in self.english_irregulars:
            return self.english_irregulars[word]

        if len(word) < 3:  # Too short to lemmatize
            return word
            
        # Handle standard verb/noun inflections
        if word.endswith('ing') and len(word) > 4:
            stem = word[:-3]
            if stem and stem[-1] == stem[-2]:  # running -> run
                return stem[:-1]
            return stem + 'e' if stem.endswith(('ak', 'tak', 'mak')) else stem
            
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'  # cities -> city
            
        if word.endswith('es') and len(word) > 3:
            if word[-3] in 'sxzh':  # boxes -> box
                return word[:-2]
            return word[:-1]
            
        if word.endswith('ed') and len(word) > 3:
            stem = word[:-2]
            if stem and stem[-1] == stem[-2]:  # planned -> plan
                return stem[:-1]
            return stem + 'e' if stem.endswith(('ak', 'tak', 'mak')) else stem
            
        if word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
            return word[:-1]  # cats -> cat
            
        return word

    def lemmatize_telugu(self, word: str) -> str:
        word = word.lower()
        
        # Process verb endings first (most specific)
        for ending, replacement in sorted(
            self.telugu_verb_endings.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):
            if word.endswith(ending):
                return word[:-len(ending)] + replacement

        # Then process noun endings
        for ending, replacement in sorted(
            self.telugu_noun_endings.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):
            if word.endswith(ending):
                return word[:-len(ending)] + replacement
        
        return word

    def lemmatize(self, word: str) -> str:
        if not word or len(word) < 2:
            return word
            
        word = word.lower().strip()
        
        # Apply appropriate lemmatization based on language
        if self.is_telugu_word(word):
            return self.lemmatize_telugu(word)
        return self.lemmatize_english(word)
# Create a singleton instance
_lemmatizer = TeluguEnglishLemmatizer()

def lemmatize_word(word: str) -> str:
    return _lemmatizer.lemmatize(word)