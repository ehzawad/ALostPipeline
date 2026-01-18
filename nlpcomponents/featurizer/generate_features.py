
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nlpcomponents.featurizer import FEATURES_DIR
from nlpcomponents.featurizer.ngram_extractor import STSNgramExtractor
from nlpcomponents.utils.constants import NGRAM_TYPES
from nlpcomponents.featurizer.feature_analyzer import STSFeatureAnalyzer
from nlpcomponents.featurizer.clean_ngrams import STSNgramCleaner, get_confused_pairs_from_analysis
from nlpcomponents.build.fingerprint import compute_dataset_fingerprint

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate STS n-gram features with optional overlap cleanup"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of top n-grams to extract per tag (default: 40)"
    )
    parser.add_argument(
        "--auto-clean",
        action="store_true",
        help="Automatically clean shared n-grams using dominance filtering"
    )
    parser.add_argument(
        "--dominance-ratio",
        type=float,
        default=2.0,
        help="Minimum frequency ratio for dominance (default: 2.0)"
    )
    parser.add_argument(
        "--min-ngrams",
        type=int,
        default=5,
        help="Minimum n-grams to keep per tag (safety net, default: 5)"
    )
    parser.add_argument(
        "--use-tfidf",
        action="store_true",
        help="Apply TF-IDF weighting to re-rank n-grams by discriminativeness"
    )
    parser.add_argument(
        "--strict-confused-pairs",
        action="store_true",
        help="Apply stricter dominance filtering for top confused tag pairs"
    )
    parser.add_argument(
        "--strict-dominance-ratio",
        type=float,
        default=4.0,
        help="Dominance ratio for confused pairs (default: 4.0)"
    )
    parser.add_argument(
        "--num-confused-pairs",
        type=int,
        default=10,
        help="Number of top confused pairs to apply strict filtering (default: 10)"
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force overwrite manual_ngrams.json even if it exists"
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("nlpcomponents/datasets/question_tag.csv"),
        help="Path to training CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nlpcomponents/datasets/features"),
        help="Directory to save feature files"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("="*70)
    print("STS N-GRAM FEATURE GENERATION")
    print("="*70)
    print(f"Top-K per tag: {args.top_k}")
    print(f"Auto-clean: {args.auto_clean}")
    if args.auto_clean:
        print(f"Dominance ratio: {args.dominance_ratio}x")
        print(f"Min n-grams per tag: {args.min_ngrams}")
        if args.use_tfidf:
            print(f"TF-IDF weighting: Enabled")
        if args.strict_confused_pairs:
            print(f"Strict confused pairs: Enabled (ratio: {args.strict_dominance_ratio}x, top {args.num_confused_pairs} pairs)")
    print(f"Force overwrite manual: {args.force_overwrite}")
    print("="*70)

    train_csv_path = args.train_csv.resolve()
    features_dir = args.output_dir.resolve()
    features_dir.mkdir(parents=True, exist_ok=True)

    auto_ngrams_path = features_dir / "auto_ngrams.json"
    manual_ngrams_path = features_dir / "manual_ngrams.json"
    overlap_analysis_path = features_dir / "overlap_analysis.json"
    cleanup_report_path = features_dir / "cleanup_report.json"

    if not train_csv_path.exists():
        print(f"\nERROR: Training data not found at {train_csv_path}")
        print(f"   Please check your path: {train_csv_path.parent}")
        sys.exit(1)

    try:
        print("\n" + "="*70)
        print("STEP 1: EXTRACTING N-GRAMS")
        print("="*70)

        print(f"Computing dataset fingerprint for {train_csv_path.name}...")
        try:
            dataset_fp = compute_dataset_fingerprint(train_csv_path)
            print(f"   Dataset fingerprint: {dataset_fp[:16]}...")
        except Exception as e:
            print(f"   Warning: Could not compute dataset fingerprint: {e}")
            dataset_fp = None

        extractor = STSNgramExtractor(
            train_csv_path=train_csv_path,
            top_k=args.top_k
        )

        extractor.generate_and_save(
            auto_path=auto_ngrams_path,
            manual_path=manual_ngrams_path,
            overwrite_manual=args.force_overwrite,
            dataset_fingerprint=dataset_fp
        )

        print(f"\nStep 1 complete!")
        print(f"   Auto file: {auto_ngrams_path}")
        print(f"   Manual file: {manual_ngrams_path}")

        print("\n" + "="*70)
        print("STEP 2: ANALYZING WITHIN-CLUSTER OVERLAPS")
        print("="*70)

        analyzer = STSFeatureAnalyzer(features_path=auto_ngrams_path)
        analysis = analyzer.analyze_and_save(output_path=overlap_analysis_path)

        print(f"\nStep 2 complete!")
        print(f"   Overlap analysis: {overlap_analysis_path}")

        if args.auto_clean:
            print("\n" + "="*70)
            print("STEP 3: CLEANING SHARED N-GRAMS (WITHIN-CLUSTER)")
            print("="*70)

            confused_pairs = None
            if args.strict_confused_pairs:
                print(f"\nExtracting top {args.num_confused_pairs} confused tag pairs...")
                confused_pairs = get_confused_pairs_from_analysis(
                    overlap_analysis_path, 
                    top_k=args.num_confused_pairs
                )
                print(f"   Found {len(confused_pairs)} confused pairs:")
                for t1, t2 in confused_pairs[:5]:
                    print(f"      {t1} <-> {t2}")
                if len(confused_pairs) > 5:
                    print(f"      ... and {len(confused_pairs) - 5} more")

            cleaner = STSNgramCleaner(
                auto_ngrams_path=auto_ngrams_path,
                overlap_analysis_path=overlap_analysis_path,
                dominance_ratio=args.dominance_ratio,
                min_ngrams_per_tag=args.min_ngrams,
                use_tfidf=args.use_tfidf,
                strict_dominance_ratio=args.strict_dominance_ratio,
                confused_pairs=confused_pairs
            )

            cleaned_features, report = cleaner.clean_and_save(
                manual_ngrams_path=manual_ngrams_path,
                cleanup_report_path=cleanup_report_path
            )

            print(f"\nStep 3 complete!")
            print(f"   Cleaned manual file: {manual_ngrams_path}")
            print(f"   Cleanup report: {cleanup_report_path}")

        else:
            print("\nStep 3: Skipped (use --auto-clean to enable)")

        print("\n" + "="*70)
        print("GENERATION COMPLETE!")
        print("="*70)

        print(f"\nGenerated files:")
        print(f"   1. {auto_ngrams_path}")
        print(f"      -> Auto-generated features (always regenerated)")
        print(f"   2. {manual_ngrams_path}")
        print(f"      -> Manual features (edit this for customization)")
        print(f"   3. {overlap_analysis_path}")
        print(f"      -> Overlap analysis")

        if args.auto_clean:
            print(f"   4. {cleanup_report_path}")
            print(f"      -> Cleanup report (dominance-based filtering)")

        print(f"\nFeature summary:")
        print(f"   Tags: {len(extractor.tags_sorted)}")
        print(f"   Top-K per tag: {args.top_k}")
        print(f"   Total n-grams: {len(extractor.tags_sorted) * args.top_k * NGRAM_TYPES}")
        print(f"   Pattern dimension: {len(extractor.tags_sorted) * NGRAM_TYPES} ({NGRAM_TYPES} features x {len(extractor.tags_sorted)} tags)")

        if args.auto_clean:
            summary = report["summary"]
            print(f"\nCleanup results:")
            for nt in ["trigrams", "fourgrams"]:
                if nt in summary:
                    stats = summary[nt]
                    print(f"   {nt.capitalize()} removed: {stats['removed']}/{stats['original']} ({stats['removed_pct']:.1f}%)")
            
            if args.use_tfidf:
                print(f"\n   TF-IDF weighting: Applied to re-rank n-grams")
            if args.strict_confused_pairs:
                print(f"   Strict filtering: Applied to {len(confused_pairs)} confused pairs")

        print(f"\nNext steps:")
        print(f"   1. Review {manual_ngrams_path} and customize if needed")
        print(f"   2. Train classifier with: python -m nlpcomponents.cli train-classifier --force")
        print(f"   3. Evaluate: python -m nlpcomponents.cli eval --data nlpcomponents/datasets/eval.csv --top-k 1")

        print("\n" + "="*70)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(
            f"\nN-gram features file not found: {e.filename}\n"
            f"Generate it by running: python -m nlpcomponents.cli features\n"
        )
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR during feature generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
