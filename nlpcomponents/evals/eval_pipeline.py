
from __future__ import annotations

import argparse
import csv
from loguru import logger
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlpcomponents import NLPPipeline

def setup_logging(debug: bool = False):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pipelineNLP accuracy on a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        default="nlpcomponents/datasets/sts_eval.csv",
        help="CSV file with columns: question, tag (default: nlpcomponents/datasets/sts_eval.csv)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many fused candidates to consider for top-k accuracy (default: 1)"
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to emit per-row evaluation details as CSV (default: disabled)"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip artifact check and rebuild (use with caution - may use stale models)"
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force rebuild artifacts even if fingerprints look up-to-date"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for verbose output"
    )
    return parser.parse_args()

def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)

def main():
    args = parse_args()
    
    setup_logging(debug=args.debug)
    
    data_path = _resolve_path(args.data)
    if not data_path.exists():
        logger.error(f"Eval CSV not found: {data_path}")
        raise FileNotFoundError(f"Eval CSV not found: {data_path}")

    logger.info(f"Loading evaluation data from {data_path}")
    rows = []
    try:
        with data_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "question" not in reader.fieldnames or "tag" not in reader.fieldnames:
                logger.error(f"CSV missing required columns. Found: {reader.fieldnames}")
                raise ValueError("CSV must contain 'question' and 'tag' columns")
            for line in reader:
                rows.append({"question": line["question"], "tag": line["tag"]})
    except Exception as e:
        logger.error(f"Failed to load eval data: {e}")
        raise

    if not rows:
        logger.warning("No rows found in eval CSV.")
        print("No rows found in eval CSV.")
        return
    
    logger.info(f"Loaded {len(rows)} evaluation rows")

    if args.skip_check:
        logger.warning("Skipping artifact check (--skip-check). May use stale artifacts!")
        print("\n[WARNING] Skipping artifact check. Results may use stale models.\n")
    else:
        logger.info("Checking and rebuilding artifacts...")
        print("\n[AUTO-REBUILD] Checking artifacts...")

        try:
            from nlpcomponents.build.orchestrator import BuildOrchestrator

            orchestrator = BuildOrchestrator(
                verbose=True,
                inference_only=True,
                force_rebuild=args.force_build
            )
            plan = orchestrator.create_build_plan()
            orchestrator.print_status_report(plan)

            result = orchestrator.build_all(force=args.force_build, dry_run=False)

            if not result.success:
                logger.error("Auto-rebuild failed")
                print("\n[ERROR] Auto-rebuild failed:")
                for artifact, error in result.failed_artifacts.items():
                    print(f"  - {artifact}: {error}")
                print("\nTry manual rebuild: python -m nlpcomponents.cli build --force")
                sys.exit(1)

            if result.rebuilt_artifacts:
                logger.info(f"Built {len(result.rebuilt_artifacts)} artifacts")
                print(f"\n[OK] Built {len(result.rebuilt_artifacts)} artifacts\n")
            else:
                logger.info("All artifacts were already up-to-date")
                print("\n[OK] All artifacts already up-to-date\n")

        except Exception as e:
            logger.error(f"Auto-rebuild failed: {e}")
            print(f"\n[ERROR] Auto-rebuild failed: {e}")
            print("Please check the logs and fix the issue manually.")
            sys.exit(1)

    logger.info("Initializing NLPPipeline...")
    try:
        pipeline = NLPPipeline()
        pipeline.initialize()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

    total = len(rows)
    processed = 0
    errors = 0
    top1_correct = 0
    topk_correct = 0
    misses_top1 = []
    misses_topk = []

    report_rows = []

    logger.info(f"Starting evaluation on {total} rows...")
    for idx, row in enumerate(rows, 1):
        question = row["question"]
        expected_tag = row["tag"]
        
        try:
            result = pipeline.run(question, fusion_top_k=args.top_k)
        except Exception as e:
            logger.error(f"Pipeline error on row {idx}: {e}")
            logger.debug(f"  Question: {question[:100]}...")
            errors += 1
            continue
            
        processed += 1
        candidates = result.get("candidates", [])
        dropped_candidates = result.get("dropped_candidates", [])
        predicted_tag = result.get("primary_tag")

        mapped_question = ""
        cosine_similarity = None
        classifier_confidence = None
        prediction_source = ""
        if candidates:
            best_candidate = candidates[0]
            prediction_source = best_candidate.get("source", "")

        signals = result.get("signals", {})
        sts_results = signals.get("sts", {}).get("results", [])
        if sts_results:
            top_sts = sts_results[0]
            mapped_question = top_sts.get("question", "")
            cosine_similarity = top_sts.get("similarity")

        clf_results = signals.get("classifier", {}).get("results", [])
        if clf_results:
            classifier_confidence = clf_results[0].get("confidence")

        if predicted_tag == expected_tag:
            top1_correct += 1
        else:
            misses_top1.append((question, expected_tag, predicted_tag))
            logger.debug(f"Miss #{len(misses_top1)}: expected={expected_tag}, got={predicted_tag}")

        candidate_tags = [cand["tag"] for cand in candidates[:args.top_k]]
        if expected_tag in candidate_tags:
            topk_correct += 1
        else:
            misses_topk.append((question, expected_tag, candidate_tags))

        top_k_scores = [
            f"{cand.get('final_score', ''):.6f}" if cand.get("final_score") is not None else ""
            for cand in candidates[:args.top_k]
        ]
        top_k_sources = [cand.get("source", "") for cand in candidates[:args.top_k]]

        all_fused_candidates = candidates + dropped_candidates
        all_fused_tags = [cand["tag"] for cand in all_fused_candidates]
        all_fused_scores = [
            f"{cand.get('final_score', ''):.6f}" if cand.get("final_score") is not None else ""
            for cand in all_fused_candidates
        ]
        all_fused_sources = [cand.get("source", "") for cand in all_fused_candidates]

        report_rows.append(
            {
                "input_question": question,
                "mapped_question": mapped_question,
                "expected_tag": expected_tag,
                "predicted_tag": predicted_tag or "",
                "cosine_similarity": f"{cosine_similarity:.6f}" if cosine_similarity is not None else "",
                "classifier_confidence": f"{classifier_confidence:.6f}" if classifier_confidence is not None else "",
                "prediction_source": prediction_source,
                "top1_correct": "1" if predicted_tag == expected_tag else "0",
                "topk_correct": "1" if expected_tag in candidate_tags else "0",
                "top_k_tags": "|".join(tag or "" for tag in candidate_tags),
                "top_k_scores": "|".join(top_k_scores),
                "top_k_sources": "|".join(top_k_sources),
                "all_fused_tags": "|".join(all_fused_tags),
                "all_fused_scores": "|".join(all_fused_scores),
                "all_fused_sources": "|".join(all_fused_sources),
                "num_dropped": str(len(dropped_candidates)),
            }
        )

        if processed > 0 and (idx % 100 == 0 or idx == total):
            current_top1_acc = 100.0 * top1_correct / processed
            current_topk_acc = 100.0 * topk_correct / processed
            print(f"Processed {processed}/{total} | Top-1: {current_top1_acc:.2f}% | Top-{args.top_k}: {current_topk_acc:.2f}%")

    denom = processed if processed > 0 else total
    top1_acc = 100.0 * top1_correct / denom if denom else 0.0
    topk_acc = 100.0 * topk_correct / denom if denom else 0.0

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Total rows: {total} (processed: {processed}, errors: {errors})")
    print(f"Top-1 accuracy: {top1_acc:.2f}% ({top1_correct}/{denom})")
    print(f"Top-{args.top_k} accuracy: {topk_acc:.2f}% ({topk_correct}/{denom})")
    print(f"Top-1 misses: {len(misses_top1)}")
    
    if misses_top1:
        print("\nSample misses (up to 5):")
        for q, exp, pred in misses_top1[:5]:
            print(f"  - Expected: {exp}")
            print(f"    Got: {pred}")
            print(f"    Question: {q[:80]}...")
    
    if len(misses_topk) != len(misses_top1):
        print(f"\nTop-{args.top_k} misses: {len(misses_topk)}")
    print("="*60)
    
    logger.info(f"Evaluation complete: Top-1={top1_acc:.2f}%, Top-{args.top_k}={topk_acc:.2f}%")

    if args.output_csv:
        output_path = _resolve_path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("w", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "input_question",
                    "mapped_question",
                    "expected_tag",
                    "predicted_tag",
                    "cosine_similarity",
                    "classifier_confidence",
                    "prediction_source",
                    "top1_correct",
                    "topk_correct",
                    "top_k_tags",
                    "top_k_scores",
                    "top_k_sources",
                    "all_fused_tags",
                    "all_fused_scores",
                    "all_fused_sources",
                    "num_dropped",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report_rows)
            print(f"\n[OUTPUT] Wrote detailed results to {output_path}")
            logger.info(f"Results written to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write output CSV: {e}")
            print(f"\n[ERROR] Failed to write output CSV: {e}")

if __name__ == "__main__":
    main()
