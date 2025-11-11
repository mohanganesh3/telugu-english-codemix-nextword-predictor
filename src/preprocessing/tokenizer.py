from typing import List, Set, Dict, Tuple, Optional
import re
from .preprocess_config import *

class TokenizerException(Exception):
    pass

class CodeMixedTokenizer:
    def __init__(self):
        # Special tokens with their descriptions
        self.special_tokens = {
            '<pad>': 'Padding token for batch processing',
            '<unk>': 'Unknown token for OOV words',
            '<s>': 'Start of sentence marker',
            '</s>': 'End of sentence marker',
            '<URL>': 'URL placeholder',
            '<EMAIL>': 'Email address placeholder',
            '<NUM>': 'Number placeholder'
        }
        
        self.word_boundaries = re.compile(r'[^\w\s]|\s+')
        self.sentence_end = re.compile(r'[.!?]+')
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.number_pattern = re.compile(r'^-?\d*\.?\d+$')

        self.keep_tokens = set(self.special_tokens.keys())
        
        self.contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hasn't": "has not", "haven't": "have not",
            "he'd": "he would", "he'll": "he will", "he's": "he is",
            "i'd": "i would", "i'll": "i will", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "wasn't": "was not",
            "we'd": "we would", "we'll": "we will", "we're": "we are",
            "we've": "we have", "weren't": "were not", "what's": "what is",
            "where's": "where is", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
        
    def expand_contractions(self, text: str) -> str:
        try:
            # Case-insensitive replacement
            processed_text = text.lower()
            for contraction, expansion in self.contractions.items():
                processed_text = re.sub(
                    rf'\b{contraction}\b', 
                    expansion, 
                    processed_text, 
                    flags=re.IGNORECASE
                )
            return processed_text
        except Exception as e:
            return text

    def is_valid_token(self, token: str) -> bool:
        try:
            if not token:
                return False
                
            # Always keep special tokens
            if token in self.keep_tokens:
                return True
                
            # Apply length constraints
            token_length = len(token)
            if token_length < PREPROCESSING_CONFIG['MIN_WORD_LENGTH']:
                return False
            if token_length > PREPROCESSING_CONFIG['MAX_WORD_LENGTH']:
                return False
                
            # Additional validation rules can be added here
            return True
            
        except Exception as e:
            return False

    def split_into_sentences(self, text: str) -> List[str]:
        try:
            # Handle common abbreviations to prevent false splits
            text = re.sub(r'(Mr|Mrs|Dr|Prof)\.', r'\1<POINT>', text)

            
            # Split on sentence boundaries
            sentences = self.sentence_end.split(text)
            
            # Clean and restore abbreviations
            cleaned_sentences = []
            for sentence in sentences:
                if sentence.strip():
                    # Restore abbreviation periods
                    cleaned = sentence.replace('<POINT>', '.')
                    cleaned = cleaned.strip()
                    cleaned_sentences.append(cleaned)
                    
            return cleaned_sentences
            
        except Exception as e:
            return [text]  # Return original text as single sentence on error

    def detect_language(self, token: str) -> str:
        try:
            # Convert to lowercase for consistent matching
            token_lower = token.lower()
            
            # Check for special tokens first
            if token in self.special_tokens:
                return 'SPECIAL'
                
            # Check for Telugu patterns
            if any(pattern in token_lower for pattern in TELUGU_PATTERNS):
                return 'TE'
                
            # Default to English if no Telugu patterns found
            return 'EN'
            
        except Exception as e:
            return 'EN'  # Default to English on error

    def clean_text(self, text: str) -> str:
        try:
            # Remove excess whitespace
            text = ' '.join(text.split())
            
            # Replace multiple punctuation with single
            text = re.sub(r'([!?,.]){2,}', r'\1', text)
            
            # Normalize quotes
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'['']', "'", text)
            
            return text
        except Exception as e:
            return text

    def detect_special_token(self, text: str) -> Optional[str]:
        text = text.strip()
        
        # URL detection
        if self.url_pattern.match(text):
            return '<URL>'
            
        # Email detection
        if self.email_pattern.match(text):
            return '<EMAIL>'
            
        # Number detection
        if self.number_pattern.match(text):
            return '<NUM>'
            
        return None

    def tokenize(self, text: str) -> List[str]:
        try:
            if not text:
                return []

            # Initial cleaning
            text = self.clean_text(text)
            
            # Add sentence boundaries
            text = f"<s> {text} </s>"
            
            # Expand contractions
            text = self.expand_contractions(text)
            
            tokens = []
            for part in self.word_boundaries.split(text):
                part = part.strip().lower()
                
                if not part:
                    continue
                    
                # Check for special tokens
                special_token = self.detect_special_token(part)
                if special_token:
                    tokens.append(special_token)
                    continue
                
                # Handle regular tokens
                if self.is_valid_token(part):
                    if part in self.special_tokens:
                        tokens.append(part)
                    else:
                        # Detect language and add token
                        lang = self.detect_language(part)
                        tokens.append(part if part != '' else '<unk>')

            # Handle maximum length
            if len(tokens) > PREPROCESSING_CONFIG['MAX_WORDS_IN_SENTENCE']:
                # Preserve special tokens at boundaries
                middle_tokens = tokens[1:-1][:PREPROCESSING_CONFIG['MAX_WORDS_IN_SENTENCE']-2]
                tokens = [tokens[0]] + middle_tokens + [tokens[-1]]

            return tokens
            
        except Exception as e:
            raise TokenizerException(f"Failed to tokenize text: {str(e)}")
    
    def is_special_token(self, token: str) -> bool:
        return token in self.special_tokens
        
    def get_vocab(self) -> Set[str]:
        return set(self.special_tokens.keys())
        
    def get_statistics(self) -> Dict[str, int]:
        return {
            'num_special_tokens': len(self.special_tokens),
            'min_length': PREPROCESSING_CONFIG['MIN_WORD_LENGTH'],
            'max_length': PREPROCESSING_CONFIG['MAX_WORD_LENGTH'],
            'max_sentence_length': PREPROCESSING_CONFIG['MAX_WORDS_IN_SENTENCE']
        }

_tokenizer_instance = None

def get_tokenizer() -> CodeMixedTokenizer:
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = CodeMixedTokenizer()
    return _tokenizer_instance

def tokenize(text: str) -> List[str]:
    return get_tokenizer().tokenize(text)