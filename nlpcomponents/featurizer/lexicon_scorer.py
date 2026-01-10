from __future__ import annotations

import json
import math
from loguru import logger
from pathlib import Path
from typing import Dict, List, Set

from ..utils.ngram_utils import extract_ngram_words

class LexiconScorer:
    
    NGRAM_WEIGHTS = {
        "unigrams": 0.10,
        "bigrams": 0.15,
        "trigrams": 0.25,
        "fourgrams": 0.25,
        "fivegrams": 0.25,
    }

    TFIDF_SOFT_SCALE: float = 1.5
    
    NGRAM_SIZES = {
        "unigrams": 1,
        "bigrams": 2,
        "trigrams": 3,
        "fourgrams": 4,
        "fivegrams": 5,
    }

    def __init__(self, ngrams_path: Path):
        weight_sum = sum(self.NGRAM_WEIGHTS.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"NGRAM_WEIGHTS must sum to 1.0, got {weight_sum}")

        self.ngrams_path = Path(ngrams_path)
        
        if not self.ngrams_path.exists():
            raise FileNotFoundError(f"N-grams file not found: {self.ngrams_path}")
        
        logger.info(f"Loading lexicon from {self.ngrams_path}")
        
        with open(self.ngrams_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.tag_patterns: Dict[str, Dict[str, Set[str]]] = {}
        self.tag_tfidf: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        for tag_name, tag_data in data.get("tags", {}).items():
            self.tag_patterns[tag_name] = {}
            self.tag_tfidf[tag_name] = {}
            
            for ng_type in self.NGRAM_WEIGHTS.keys():
                items = tag_data.get(ng_type, [])
                self.tag_patterns[tag_name][ng_type] = set(
                    item["ngram"] for item in items
                )
                self.tag_tfidf[tag_name][ng_type] = {
                    item["ngram"]: item.get("tfidf_score", 1.0) 
                    for item in items
                }
        
        self.tags = sorted(self.tag_patterns.keys())
        logger.info(f"  Loaded lexicon patterns for {len(self.tags)} tags")
    
    def _compute_tag_score(self, tag: str, query_ngrams: Dict[int, Set[str]]) -> float:
        if tag not in self.tag_patterns:
            logger.warning(
                f"Tag '{tag}' not found in lexicon patterns. "
                "This may indicate stale artifacts: regenerate with 'python -m nlpcomponents.cli features --force'"
            )
            return 0.0
        
        total_score = 0.0
        
        for ng_type, weight in self.NGRAM_WEIGHTS.items():
            n = self.NGRAM_SIZES[ng_type]
            q_ngrams = query_ngrams.get(n, set())
            
            if not q_ngrams:
                continue
            
            tag_ngrams = self.tag_patterns[tag].get(ng_type, set())
            matches = q_ngrams & tag_ngrams
            
            if not matches:
                continue
            
            tfidf_dict = self.tag_tfidf[tag].get(ng_type, {})
            tfidf_sum = sum(tfidf_dict.get(m, 1.0) for m in matches)

            avg_tfidf = tfidf_sum / len(matches)
            normalized = math.tanh(avg_tfidf / self.TFIDF_SOFT_SCALE)
            total_score += weight * normalized

        return total_score
    
    def score(self, question: str) -> Dict[str, float]:
        return self.score_candidates(question, self.tags)
    
    def score_candidates(
        self,
        question: str,
        candidate_tags: List[str]
    ) -> Dict[str, float]:
        if not question or not question.strip():
            return {tag: 0.0 for tag in candidate_tags}

        query_ngrams = {
            n: extract_ngram_words(question, n)
            for n in self.NGRAM_SIZES.values()
        }
        
        scores = {}
        for tag in candidate_tags:
            scores[tag] = self._compute_tag_score(tag, query_ngrams)
        
        return scores
    
    def get_matching_ngrams(
        self, 
        question: str, 
        tag: str
    ) -> Dict[str, List[str]]:
        if tag not in self.tag_patterns:
            return {}
        
        matches = {}
        for ng_type, n in self.NGRAM_SIZES.items():
            q_ngrams = extract_ngram_words(question, n)
            tag_ngrams = self.tag_patterns[tag].get(ng_type, set())
            matched = q_ngrams & tag_ngrams
            if matched:
                matches[ng_type] = sorted(matched)
        
        return matches
