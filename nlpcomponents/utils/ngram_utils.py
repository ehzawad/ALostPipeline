from __future__ import annotations

import re
import threading
from typing import List, Set, Optional, Callable
from loguru import logger

_tokenizer_lock = threading.RLock()
_subword_tokenizer: Optional[Callable[[str], List[str]]] = None
_tokenizer_mode: str = "whitespace"

def configure_tokenizer(mode: str = "whitespace", model_name: Optional[str] = None) -> None:
    global _subword_tokenizer, _tokenizer_mode
    with _tokenizer_lock:
        if mode == "whitespace":
            _subword_tokenizer = None
            _tokenizer_mode = "whitespace"
        elif mode == "sentencepiece":
            import sentencepiece as spm
            if model_name is None:
                raise ValueError("model_name required for sentencepiece mode")
            sp = spm.SentencePieceProcessor()
            sp.Load(model_name)
            _subword_tokenizer = lambda text: sp.EncodeAsPieces(text.lower())
            _tokenizer_mode = "sentencepiece"
        elif mode == "wordpiece":
            from transformers import AutoTokenizer
            model_name = model_name or "bert-base-uncased"
            wp_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _subword_tokenizer = lambda text: wp_tokenizer.tokenize(text.lower())
            _tokenizer_mode = "wordpiece"
        else:
            raise ValueError(f"Unknown tokenizer mode: {mode}")

def tokenize_words(text: str) -> List[str]:
    with _tokenizer_lock:
        tokenizer = _subword_tokenizer
        current_mode = _tokenizer_mode

    if tokenizer is not None and current_mode != "whitespace":
        try:
            tokens = tokenizer(text)
            return [t for t in tokens if t and not t.startswith('[') and not t.startswith('<')]
        except Exception as e:
            logger.warning(
                f"Subword tokenizer ({current_mode}) failed, falling back to whitespace: {e}"
            )

    words = text.lower().split()
    stripped = []
    for w in words:
        w = re.sub(r'^[^\w]+|[^\w]+$', '', w, flags=re.UNICODE)
        if w and re.search(r'\w', w, re.UNICODE):
            stripped.append(w)
    return stripped

def get_tokenizer_mode() -> str:
    with _tokenizer_lock:
        return _tokenizer_mode

def extract_ngram_words(text: str, n: int) -> Set[str]:
    words = tokenize_words(text)
    if len(words) < n:
        return set()
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}

def extract_ngram_words_list(text: str, n: int) -> List[str]:
    words = tokenize_words(text)
    if len(words) < n:
        return []
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def extract_ngrams_from_tokens(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def extract_ngrams_from_tokens_set(tokens: List[str], n: int) -> Set[str]:
    if len(tokens) < n:
        return set()
    return {' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def extract_all_ngrams(text: str, max_n: int = 5) -> dict:
    ngram_names = ['unigrams', 'bigrams', 'trigrams', 'fourgrams', 'fivegrams']
    result = {}
    for i, name in enumerate(ngram_names[:max_n], start=1):
        result[name] = extract_ngram_words(text, i)
    return result
