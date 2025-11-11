# -*- coding: utf-8 -*-
"""
Stop words for Telugu-English code-mixed text processing.
Includes both English stop words and romanized Telugu stop words.
"""

# Common English stop words
ENGLISH_STOP_WORDS = {
    # Articles
    'a', 'an', 'the',
    
    # Pronouns
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
    'who', 'whom', 'whose',
    'what', 'which',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'up', 'down', 'into', 'onto', 'upon',
    'about', 'above', 'across', 'after', 'against',
    'along', 'among', 'around', 'before', 'behind',
    'below', 'beneath', 'beside', 'between', 'beyond',
    
    # Conjunctions
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
    'although', 'because', 'since', 'unless',
    
    # Auxiliary verbs
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might',
    'must', 'can', 'could',
    
    # Other common words
    'then', 'than', 'there', 'here', 'where', 'when',
    'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'not', 'only', 'own', 'same', 'so',
    'too', 'very', 'just', 'now'
}

# Romanized Telugu stop words
TELUGU_STOP_WORDS = {
    # Pronouns
    'nenu', 'naaku', 'naa',           # I, me, my
    'meeru', 'meeku', 'mee',          # You (formal), your
    'nuvvu', 'neeku', 'nee',          # You (informal), your
    'atanu', 'atani', 'atanidi',      # He, him, his
    'aame', 'aamenu', 'aamedi',       # She, her, hers
    'adi', 'daani', 'daanidi',        # It, its
    'manamu', 'mana', 'manaki',       # We, our, us
    'vaaru', 'vaallu', 'vaalla',      # They, them, their
    
    # Demonstratives
    'idi', 'adi', 'ivi', 'avi',      # This, that, these, those
    
    # Question words
    'emi', 'enduku', 'ela',           # What, why, how
    'ekkada', 'eppudu', 'evaru',      # Where, when, who
    
    # Postpositions
    'lo', 'ki', 'pai', 'tho',         # In, to, on, with
    'nunchi', 'dwara', 'valla',       # From, through, because of
    'kosam', 'gurinchi', 'madhya',    # For, about, between
    
    # Conjunctions
    'mariyu', 'kani', 'leda',         # And, but, or
    'kaani', 'ayite', 'ante',         # But, then, means
    
    # Common particles
    'oo', 'gaa', 'le', 'ra', 'andi',  # Common particles
    'kada', 'mari', 'anta',           # Question/emphasis particles
    
    # Other common words
    'ala', 'ila', 'kuda', 'inkaa',    # Like that, like this, also, more
    'inka', 'malli', 'ippudu',        # More, again, now
    'akkada', 'ikkada', 'appudu',     # There, here, then
    'ippatiki', 'mundhu', 'tarvata'   # Until now, before, after
}

# Combined stop words for code-mixed text
STOP_WORDS = ENGLISH_STOP_WORDS | TELUGU_STOP_WORDS