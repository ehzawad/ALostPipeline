from __future__ import annotations

import argparse
import json
from loguru import logger
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlpcomponents.build.fingerprint import compute_dataset_fingerprint
from nlpcomponents.utils.ngram_utils import extract_ngram_words_list
from nlpcomponents.utils.json_utils import json_default

class VocabularyBuilder:

    def __init__(
        self,
        train_csv_path: Path,
        min_count: int = 3,
        exclude_tags: Optional[Set[str]] = None
    ):
        self.train_csv_path = train_csv_path
        self.min_count = max(1, min_count)
        self.exclude_tags = exclude_tags or set()

        logger.info("Using word-level tokenization (language-agnostic)")

        self.word_unigrams: Set[str] = set()
        self.word_bigrams: Set[str] = set()
        self.word_trigrams: Set[str] = set()
        self.word_fourgrams: Set[str] = set()
        self.word_fivegrams: Set[str] = set()

        self.word_unigram_counts: Counter[str] = Counter()
        self.word_bigram_counts: Counter[str] = Counter()
        self.word_trigram_counts: Counter[str] = Counter()
        self.word_fourgram_counts: Counter[str] = Counter()
        self.word_fivegram_counts: Counter[str] = Counter()

    def process_questions(self, questions: pd.Series) -> None:
        logger.info(f"Processing {len(questions)} questions...")

        for question in tqdm(questions, desc="Extracting n-grams", disable=False):
            if pd.isna(question):
                continue
            question_str = str(question)

            for n in range(1, 6):
                word_ngrams = extract_ngram_words_list(question_str, n)
                if not word_ngrams:
                    continue
                if n == 1:
                    self.word_unigram_counts.update(word_ngrams)
                elif n == 2:
                    self.word_bigram_counts.update(word_ngrams)
                elif n == 3:
                    self.word_trigram_counts.update(word_ngrams)
                elif n == 4:
                    self.word_fourgram_counts.update(word_ngrams)
                elif n == 5:
                    self.word_fivegram_counts.update(word_ngrams)

        self.word_unigrams = {ng for ng, c in self.word_unigram_counts.items() if c >= self.min_count}
        self.word_bigrams = {ng for ng, c in self.word_bigram_counts.items() if c >= self.min_count}
        self.word_trigrams = {ng for ng, c in self.word_trigram_counts.items() if c >= self.min_count}
        self.word_fourgrams = {ng for ng, c in self.word_fourgram_counts.items() if c >= self.min_count}
        self.word_fivegrams = {ng for ng, c in self.word_fivegram_counts.items() if c >= self.min_count}

    def build(self) -> Dict[str, Any]:
        logger.info(f"Loading training data from {self.train_csv_path}")
        df = pd.read_csv(self.train_csv_path)

        logger.info(f"  Loaded {len(df)} samples")
        logger.info(f"  Unique tags: {df['tag'].nunique()}")

        if self.exclude_tags:
            dataset_tags = set(df['tag'].unique())
            unknown_excludes = self.exclude_tags - dataset_tags
            if unknown_excludes:
                logger.warning(
                    f"  Exclude tags not found in dataset (ignored): {sorted(unknown_excludes)}"
                )
            
            before = len(df)
            df = df[~df["tag"].isin(self.exclude_tags)]
            excluded_count = before - len(df)
            logger.info(
                f"  Excluded tags {sorted(self.exclude_tags & dataset_tags)} -> "
                f"removed {excluded_count} rows, kept {len(df)} of {before}"
            )

        self.process_questions(df['question'])

        dataset_fp = compute_dataset_fingerprint(
            self.train_csv_path,
            columns=('question', 'tag')
        )

        result = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "training_samples": len(df),
                "num_tags": df['tag'].nunique(),
                "tokenizer": "word-level (language-agnostic)",
                "config": {
                    "min_count": self.min_count,
                    "exclude_tags": sorted(self.exclude_tags)
                },
                "dependencies": {
                    "dataset": {
                        "fingerprint": dataset_fp,
                        "file": "question_tag.csv",
                        "num_samples": len(df),
                        "num_tags": df['tag'].nunique()
                    }
                }
            },
            "word_unigrams": sorted(list(self.word_unigrams)),
            "word_bigrams": sorted(list(self.word_bigrams)),
            "word_trigrams": sorted(list(self.word_trigrams)),
            "word_fourgrams": sorted(list(self.word_fourgrams)),
            "word_fivegrams": sorted(list(self.word_fivegrams)),
            "statistics": {
                "total_word_unigrams": len(self.word_unigrams),
                "total_word_bigrams": len(self.word_bigrams),
                "total_word_trigrams": len(self.word_trigrams),
                "total_word_fourgrams": len(self.word_fourgrams),
                "total_word_fivegrams": len(self.word_fivegrams),
                "min_count": self.min_count,
                "exclude_tags": sorted(self.exclude_tags)
            }
        }

        logger.info(f"\nVocabulary statistics:")
        logger.info(f"  Unigrams: {len(self.word_unigrams):,}")
        logger.info(f"  Bigrams: {len(self.word_bigrams):,}")
        logger.info(f"  Trigrams: {len(self.word_trigrams):,}")
        logger.info(f"  Fourgrams: {len(self.word_fourgrams):,}")
        logger.info(f"  Fivegrams: {len(self.word_fivegrams):,}")

        return result

def build_vocabulary(
    train_csv: Path,
    output_path: Path,
    min_count: int = 3,
    exclude_tags: Optional[Set[str]] = None
) -> Dict[str, Any]:
    try:
        builder = VocabularyBuilder(
            train_csv,
            min_count=min_count,
            exclude_tags=exclude_tags
        )
        vocab_data = builder.build()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2, default=json_default)

        logger.info(f"\nVocabulary saved to {output_path}")

        return {
            'success': True,
            'stats': vocab_data['statistics'],
            'output_path': str(output_path)
        }

    except Exception as e:
        logger.error(f"Failed to build vocabulary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(
        description="Build training vocabulary for OOV detection"
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ROOT_DIR / "nlpcomponents" / "datasets" / "question_tag.csv",
        help="Path to question_tag.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "nlpcomponents" / "datasets" / "features" / "training_vocabulary.json",
        help="Output path for vocabulary JSON"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum frequency required to keep a token (default: 3)"
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="Tag to exclude from vocabulary"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING VOCABULARY BUILDER")
    print("=" * 80)
    print(f"\nInput: {args.train_csv}")
    print(f"Output: {args.output}")
    print(f"Tokenizer: word-level (language-agnostic)")
    print(f"Min count: {args.min_count}")
    print(f"Exclude tags: {sorted(set(args.exclude_tag or []))}")
    print()

    result = build_vocabulary(
        train_csv=args.train_csv,
        output_path=args.output,
        min_count=args.min_count,
        exclude_tags=set(args.exclude_tag or [])
    )

    print("\n" + "=" * 80)
    if result['success']:
        print("BUILD SUCCESSFUL")
        print("=" * 80)
        stats = result['stats']
        print(f"\nWord-level Vocabulary (OOV detection):")
        print(f"  Unigrams: {stats['total_word_unigrams']:,}")
        print(f"  Bigrams: {stats['total_word_bigrams']:,}")
        print(f"  Trigrams: {stats['total_word_trigrams']:,}")
        print(f"  Fourgrams: {stats['total_word_fourgrams']:,}")
        print(f"  Fivegrams: {stats['total_word_fivegrams']:,}")
        print(f"\nSaved to: {result['output_path']}")
        sys.exit(0)
    else:
        print("BUILD FAILED")
        print("=" * 80)
        print(f"\nError: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
