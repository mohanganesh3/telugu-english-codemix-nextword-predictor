import re

PREPROCESSING_CONFIG = {
    "LANGUAGE_CODES": {"TE", "EN"},
    "MIN_WORD_LENGTH": 2,
    "MAX_WORD_LENGTH": 50,
    "MIN_SENTENCE_LENGTH": 3,
    "MAX_SENTENCE_LENGTH": 100,
    "MIN_WORDS_IN_SENTENCE": 2,
    "MAX_WORDS_IN_SENTENCE": 50
}

TELUGU_PATTERNS = [
    'aa', 'ee', 'oo', 'th', 'ch', 'dh', 'gh', 
    'kh', 'ph', 'bh', 'sh', 'ng', 'nk', 'mb',
    'nnu', 'nna', 'ndi', 'nchi', 'tha', 'dhi',
    'avu', 'anu', 'ali', 'aku', 'ata', 'ani',
    'la', 'ra', 'ta', 'da', 'pa', 'ba', 'ma', 'na', 'sa', 'va', 'ka', 'ga', 'ja', 'cha', 'sha',
    'pu', 'bu', 'mu', 'nu', 'ru', 'lu', 'ku', 'gu', 'du', 'tu', 'su', 'vu',
    'mee', 'nee', 'che', 'the', 'dhe', 'kku', 'ppu', 'llu', 'nni', 'tti', 'ddi',
    'ram', 'nam', 'vam', 'sam', 'kam', 'pam', 'tam', 'dam',
    'allu', 'illu', 'pilla', 'konda', 'giri', 'peta', 'palli', 'ooru', 'cheruvu', 'gunta',
    'vadu', 'vallu', 'vadiki', 'vadini', 'vadilo', 'vaditho', 'vadinunchi',
    'ante', 'kada', 'ledu', 'undhi', 'unna', 'vachu', 'poyindi', 'ochindi', 'vellipoyindi',
    'padutondi', 'cheppandi', 'chudandi', 'tisukondi', 'pampandi', 'pettu', 'tisuko', 'tisukondi'
]


TELUGU_VERB_ENDINGS = {
    # Present continuous
    'tunnaanu': 'u',   # 1st person
    'tunnavu': 'u',    # 2nd person
    'tunnaadu': 'u',   # 3rd person masculine
    'tundi': 'u',      # 3rd person neutral
    # Future tense
    'thaanu': 'u',     # 1st person
    'thaavu': 'u',     # 2nd person
    'thaadu': 'u',     # 3rd person masculine
    'thundi': 'u',     # 3rd person neutral
    # Past tense
    'aanu': 'u',       # 1st person
    'aavu': 'u',       # 2nd person
    'aadu': 'u',       # 3rd person masculine
    'indi': 'u',       # 3rd person neutral
    # Perfect aspect
    'esaanu': 'u',     # 1st person
    'esaavu': 'u',     # 2nd person
    'esaadu': 'u',     # 3rd person masculine
    'esindi': 'u'      # 3rd person neutral
}

TELUGU_NOUN_ENDINGS = {
    'lu': '',        # Plural
    'ni': '',        # Accusative
    'ki': '',        # Dative
    'tho': '',       # Instrumental
    'lo': '',        # Locative
    'nunchi': '',    # Ablative
    'dwara': '',     # Instrumental
    'gaa': '',       # Adverbial
    'aina': ''       # Adjective/Past participle
}

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
NUMBER_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\b')
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')
REPEATING_CHARS_PATTERN = re.compile(r'(.)\1{2,}')