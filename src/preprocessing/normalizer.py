from typing import Dict, List, Set
import re
from .preprocess_config import *

class TextNormalizer:
    def __init__(self):
        self.telugu_variants = {
            'aa': 'a', 'ee': 'i', 'oo': 'o',
            'mm': 'm', 'nn': 'n', 'll': 'l',
            'th': 't', 'dh': 'd', 'gh': 'g',
            'bh': 'b', 'ph': 'f', 'kh': 'k'
        }
        
        self.chat_abbrevs = {
            'u': 'you', 'r': 'are', 'y': 'why', 'n': 'and',
            'k': 'ok', 'gud': 'good', 'pic': 'picture',
            'pics': 'pictures', 'plz': 'please', 'pls': 'please',
            'thx': 'thanks', 'thnx': 'thanks', 'msg': 'message',
            'tmrw': 'tomorrow', 'idk': 'i do not know',
            'tbh': 'to be honest', 'imo': 'in my opinion',
            'asap': 'as soon as possible', 'lol': 'laugh',
            'omg': 'oh my god', 'wtf': 'what the',
            'aka': 'also known as', 'etc': 'etcetera'
        }
        
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)

    def normalize_repeating_chars(self, text: str) -> str:
        return REPEATING_CHARS_PATTERN.sub(r'\1\1', text)

    def normalize_chat_spellings(self, text: str) -> str:
        words = text.split()
        return ' '.join(self.chat_abbrevs.get(word.lower(), word) for word in words)

    def normalize_telugu_variants(self, text: str) -> str:
        for variant, standard in self.telugu_variants.items():
            text = text.replace(variant, standard)
        return text

    def remove_urls_and_emails(self, text: str) -> str:
        text = URL_PATTERN.sub('<URL>', text)
        return EMAIL_PATTERN.sub('<EMAIL>', text)

    def remove_special_chars(self, text: str) -> str:
        return SPECIAL_CHARS_PATTERN.sub(' ', text)

    def remove_emojis(self, text: str) -> str:
        return self.emoji_pattern.sub('', text)

    def normalize_numbers(self, text: str) -> str:
        return NUMBER_PATTERN.sub('<NUM>', text)

    def clean_whitespace(self, text: str) -> str:
        text = WHITESPACE_PATTERN.sub(' ', text)
        return text.strip()

    def normalize_text(self, text: str) -> str:
        if not text or len(text) < PREPROCESSING_CONFIG['MIN_WORD_LENGTH']:
            return text

        text = text.lower()
        text = self.remove_urls_and_emails(text)
        text = self.remove_emojis(text)
        text = self.normalize_numbers(text)
        text = self.normalize_repeating_chars(text)
        text = self.normalize_chat_spellings(text)
        text = self.normalize_telugu_variants(text)
        text = self.remove_special_chars(text)
        text = self.clean_whitespace(text)

        if len(text) > PREPROCESSING_CONFIG['MAX_WORD_LENGTH']:
            text = text[:PREPROCESSING_CONFIG['MAX_WORD_LENGTH']]

        return text

# Create singleton instance
normalizer = TextNormalizer()

def normalize_text(text: str) -> str:
    """Convenience function that uses the singleton normalizer"""
    return normalizer.normalize_text(text)