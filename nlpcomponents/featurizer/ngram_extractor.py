import json
import math
from loguru import logger
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..utils.constants import NGRAM_TYPES
from ..utils.ngram_utils import extract_ngram_words_list

def compute_cross_tag_idf(all_ngrams_by_tag: Dict[str, Set[str]]) -> Dict[str, float]:
    num_tags = len(all_ngrams_by_tag)
    if num_tags == 0:
        return {}
    
    doc_freq: Counter = Counter()
    for ngrams in all_ngrams_by_tag.values():
        for ngram in ngrams:
            doc_freq[ngram] += 1
    
    idf: Dict[str, float] = {}
    for ngram, df in doc_freq.items():
        idf[ngram] = math.log(num_tags / df)
    
    return idf

def compute_tfidf_scores(
    tag_ngram_counts: Dict[str, Counter],
    global_idf: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    if global_idf is None:
        all_ngrams_by_tag = {
            tag: set(counts.keys())
            for tag, counts in tag_ngram_counts.items()
        }
        global_idf = compute_cross_tag_idf(all_ngrams_by_tag)
    
    tfidf_scores: Dict[str, Dict[str, float]] = {}
    
    for tag, counts in tag_ngram_counts.items():
        if not counts:
            tfidf_scores[tag] = {}
            continue

        max_freq = max(counts.values()) if counts.values() else 1
        tf: Dict[str, float] = {}
        for ngram, freq in counts.items():
            tf[ngram] = 0.5 + 0.5 * (freq / max_freq)
        
        tfidf_scores[tag] = {
            ngram: tf.get(ngram, 0) * global_idf.get(ngram, 0)
            for ngram in counts.keys()
        }
    
    return tfidf_scores

class STSNgramExtractor:

    def __init__(self, train_csv_path: Path, top_k: int = 40):
        self.train_csv_path = train_csv_path
        self.top_k = top_k
        self.train_df = None
        self.tags_sorted = None

    def load_training_data(self) -> None:
        print(f"Loading STS training data from {self.train_csv_path}")
        self.train_df = pd.read_csv(self.train_csv_path)

        print(f"Loaded {len(self.train_df)} samples")
        print(f"Columns: {list(self.train_df.columns)}")

        self.tags_sorted = sorted(self.train_df['tag'].unique())
        print(f"Unique tags: {len(self.tags_sorted)}")

        print("[OK] Training data loaded")

    def extract_tag_ngrams(self, tag: str, n: int) -> Counter:
        tag_questions = self.train_df[self.train_df['tag'] == tag]['question']

        all_ngrams = []
        for question in tag_questions:
            if question is None or (isinstance(question, float) and np.isnan(question)):
                continue
            all_ngrams.extend(extract_ngram_words_list(str(question), n))

        return Counter(all_ngrams)

    def generate_ngram_features(self, dataset_fingerprint: Optional[str] = None) -> Dict:
        num_samples = len(self.train_df)
        num_tags = len(self.tags_sorted)
        pattern_dim = num_tags * NGRAM_TYPES
        
        print(f"\nExtracting n-gram features for {num_tags} STS tags...")
        logger.info(f"Generating n-gram features: {num_samples} samples, {num_tags} tags")

        features = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "training_samples": num_samples,
                "num_tags": num_tags,
                "top_k": self.top_k,
                "ngram_types": ["unigrams", "bigrams", "trigrams", "fourgrams", "fivegrams"],
                "pattern_dim": pattern_dim,
                "dependencies": {
                    "dataset": {
                        "fingerprint": dataset_fingerprint,
                        "file": str(self.train_csv_path.name),
                        "num_samples": num_samples,
                        "num_tags": num_tags
                    }
                } if dataset_fingerprint else {}
            },
            "tags": {}
        }

        for tag in self.tags_sorted:
            print(f"Processing tag: {tag}")

            unigram_counts = self.extract_tag_ngrams(tag, 1)
            top_unigrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    unigram_counts.most_common(self.top_k)
                )
            ]

            bigram_counts = self.extract_tag_ngrams(tag, 2)
            top_bigrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    bigram_counts.most_common(self.top_k)
                )
            ]

            trigram_counts = self.extract_tag_ngrams(tag, 3)
            top_trigrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    trigram_counts.most_common(self.top_k)
                )
            ]

            fourgram_counts = self.extract_tag_ngrams(tag, 4)
            top_fourgrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    fourgram_counts.most_common(self.top_k)
                )
            ]

            fivegram_counts = self.extract_tag_ngrams(tag, 5)
            top_fivegrams = [
                {
                    "ngram": ngram,
                    "frequency": freq,
                    "rank": rank + 1
                }
                for rank, (ngram, freq) in enumerate(
                    fivegram_counts.most_common(self.top_k)
                )
            ]

            features["tags"][tag] = {
                "group": "all_tags",
                "unigrams": top_unigrams,
                "bigrams": top_bigrams,
                "trigrams": top_trigrams,
                "fourgrams": top_fourgrams,
                "fivegrams": top_fivegrams,
                "total_unigrams": len(unigram_counts),
                "total_bigrams": len(bigram_counts),
                "total_trigrams": len(trigram_counts),
                "total_fourgrams": len(fourgram_counts),
                "total_fivegrams": len(fivegram_counts)
            }

            print(f"  - Unigrams: {len(top_unigrams)} (of {len(unigram_counts)} unique)")
            print(f"  - Bigrams: {len(top_bigrams)} (of {len(bigram_counts)} unique)")
            print(f"  - Trigrams: {len(top_trigrams)} (of {len(trigram_counts)} unique)")
            print(f"  - Fourgrams: {len(top_fourgrams)} (of {len(fourgram_counts)} unique)")
            print(f"  - Fivegrams: {len(top_fivegrams)} (of {len(fivegram_counts)} unique)")

        return features

    def save_features(self, features: Dict, output_path: Path) -> None:
        print(f"\nSaving features to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        print(f"Saved successfully!")

    def generate_and_save(
        self,
        auto_path: Path,
        manual_path: Path,
        overwrite_manual: bool = False,
        dataset_fingerprint: Optional[str] = None
    ) -> Dict:
        self.load_training_data()

        features = self.generate_ngram_features(dataset_fingerprint=dataset_fingerprint)

        self.save_features(features, auto_path)

        if overwrite_manual or not manual_path.exists():
            self.save_features(features, manual_path)
            print(f"\nManual file initialized at {manual_path}")
            print("You can now edit this file to customize n-gram features per tag!")
        else:
            print(f"\nManual file already exists at {manual_path}")
            print("Not overwriting. Use overwrite_manual=True to regenerate.")

        print("\n" + "="*70)
        print("STS N-GRAM FEATURE EXTRACTION SUMMARY")
        print("="*70)
        print(f"Training samples: {len(self.train_df)}")
        print(f"Tags: {len(self.tags_sorted)}")
        print(f"Top-K per tag: {self.top_k}")
        print(f"Total n-grams: {len(self.tags_sorted) * self.top_k * NGRAM_TYPES}")
        print(f"Pattern dimension: {len(self.tags_sorted) * NGRAM_TYPES} ({NGRAM_TYPES} types × {len(self.tags_sorted)} tags)")
        print(f"Auto file: {auto_path}")
        print(f"Manual file: {manual_path}")
        print("="*70)

        return features

def load_ngram_features(features_path: Path) -> Tuple[Dict[str, Dict], List[str]]:
    print(f"Loading n-gram features from {features_path}")

    with open(features_path, 'r', encoding='utf-8') as f:
        features = json.load(f)

    tag_patterns = {}
    for tag_name, tag_data in features["tags"].items():
        tag_patterns[tag_name] = {
            "unigrams": set(item["ngram"] for item in tag_data.get("unigrams", [])),
            "bigrams": set(item["ngram"] for item in tag_data.get("bigrams", [])),
            "trigrams": set(item["ngram"] for item in tag_data.get("trigrams", [])),
            "fourgrams": set(item["ngram"] for item in tag_data.get("fourgrams", [])),
            "fivegrams": set(item["ngram"] for item in tag_data.get("fivegrams", []))
        }

    tags_sorted = sorted(tag_patterns.keys())

    print(f"Loaded features for {len(tags_sorted)} tags")
    print(f"Pattern dimension: {len(tags_sorted) * NGRAM_TYPES} ({NGRAM_TYPES} types × {len(tags_sorted)} tags)")

    for tag in tags_sorted[:3]:
        print(f"  {tag}: {len(tag_patterns[tag]['unigrams'])} unigrams, "
              f"{len(tag_patterns[tag]['bigrams'])} bigrams, "
              f"{len(tag_patterns[tag]['trigrams'])} trigrams, "
              f"{len(tag_patterns[tag]['fourgrams'])} fourgrams, "
              f"{len(tag_patterns[tag]['fivegrams'])} fivegrams")

    return tag_patterns, tags_sorted
