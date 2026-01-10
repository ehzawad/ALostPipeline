from __future__ import annotations

import unicodedata
from typing import List

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFKC', text)

def normalize_case(text: str) -> str:
    return text.casefold()

def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize('NFD', text)
    stripped = ''.join(
        char for char in decomposed
        if unicodedata.category(char) != 'Mn'
    )
    return unicodedata.normalize('NFC', stripped)

def normalize_whitespace(text: str) -> str:
    return ' '.join(text.split())

def remove_punctuation(text: str) -> str:
    return ''.join(
        char for char in text
        if not unicodedata.category(char).startswith('P')
    )

class TextNormalizer:
    
    def __init__(
        self,
        unicode_normalize: bool = True,
        case_fold: bool = True,
        strip_diacritics: bool = False,
        remove_punctuation: bool = True,
        normalize_whitespace: bool = True,
    ):
        self.unicode_normalize = unicode_normalize
        self.case_fold = case_fold
        self.strip_diacritics_enabled = strip_diacritics
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
    
    def normalize(self, text: str) -> str:
        if not text:
            return text
        
        if self.unicode_normalize:
            text = normalize_unicode(text)
        
        if self.case_fold:
            text = normalize_case(text)
        
        if self.strip_diacritics_enabled:
            text = strip_diacritics(text)
        
        if self.remove_punctuation:
            text = remove_punctuation(text)
        
        if self.normalize_whitespace:
            text = normalize_whitespace(text)
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        return [self.normalize(t) for t in texts]

DEFAULT_NORMALIZER = TextNormalizer()
