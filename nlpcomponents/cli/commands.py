from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from nlpcomponents.config import NLPPipelineConfig
from nlpcomponents.core.pipeline import NLPPipeline
from nlpcomponents.build.orchestrator import BuildOrchestrator
from nlpcomponents.build.executor import BuildExecutor
from nlpcomponents.cache.embedding_cache import EmbeddingCacheManager
from nlpcomponents.utils.path_utils import PROJECT_ROOT

def _setup_logging(debug: bool):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level
    )

def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path

def _load_config(args: argparse.Namespace) -> NLPPipelineConfig:
    config_data = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
            
    config = NLPPipelineConfig.from_dict(config_data)
    
    if args.train_csv:
        config.dataset.train_csv = _resolve(args.train_csv)
    if args.eval_csv:
        config.dataset.eval_csv = _resolve(args.eval_csv)
        
    if args.output_dir:
        out_root = _resolve(args.output_dir)
        config.output.features_dir = out_root / "features"
        config.output.classifier_dir = out_root / "models" / "tag_classifier"
        config.output.semantic_dir = out_root / "models" / "semantic"
        
        config.__post_init__()
        
    return config

def _print_or_status(plan: dict, orchestrator: BuildOrchestrator, show_status: bool):
    if show_status:
        orchestrator.print_status_report(plan)
    else:
        status = plan.get("status") or orchestrator.analyze_status()
        print(json.dumps(status, indent=2, default=str))

def cmd_check(args: argparse.Namespace):
    config = _load_config(args)
    orch = BuildOrchestrator(
        config=config,
        verbose=not args.quiet,
        inference_only=args.inference_only,
        force_rebuild=args.force,
        dry_run=args.dry_run
    )
    plan = orch.create_build_plan()
    if args.json:
        print(json.dumps(plan, indent=2, default=str))
        return
    _print_or_status(plan, orch, show_status=True)

def _execute_plan(orchestrator: BuildOrchestrator, plan: dict, verbose: bool):
    executor = BuildExecutor(config=orchestrator.config, verbose=verbose)
    ok = orchestrator.execute_plan(plan, executor)
    if not ok:
        sys.exit(1)

def cmd_build(args: argparse.Namespace):
    config = _load_config(args)
    orch = BuildOrchestrator(
        config=config,
        verbose=not args.quiet,
        inference_only=args.inference_only,
        force_rebuild=args.force,
        dry_run=args.dry_run
    )
    plan = orch.create_build_plan()
    _print_or_status(plan, orch, show_status=args.show_status)
    if args.dry_run:
        return
    _execute_plan(orch, plan, verbose=not args.quiet)

def _artifact_plan(artifacts: list[str], args: argparse.Namespace) -> tuple[BuildOrchestrator, dict]:
    config = _load_config(args)
    orch = BuildOrchestrator(
        config=config,
        verbose=not args.quiet,
        inference_only=args.inference_only,
        force_rebuild=args.force,
        dry_run=args.dry_run
    )
    build_list = orch.calculate_rebuild_set(force=args.force, artifacts=artifacts)
    plan = {
        "build_list": build_list,
        "status": orch.analyze_status()
    }
    return orch, plan

def cmd_features(args: argparse.Namespace):
    orch, plan = _artifact_plan(
        ["training_vocabulary.json", "manual_ngrams.json"],
        args
    )
    _print_or_status(plan, orch, show_status=args.show_status)
    if args.dry_run:
        return
    if not plan["build_list"] and not args.force:
        print("Features already up-to-date")
        return
    _execute_plan(orch, plan, verbose=not args.quiet)

def cmd_train_classifier(args: argparse.Namespace):
    orch, plan = _artifact_plan(["unified_tag_classifier.pth"], args)
    _print_or_status(plan, orch, show_status=args.show_status)
    if args.dry_run:
        return
    if not plan["build_list"] and not args.force:
        print("Classifier already up-to-date")
    else:
        _execute_plan(orch, plan, verbose=not args.quiet)

def cmd_train_faiss(args: argparse.Namespace):
    orch, plan = _artifact_plan(["faiss_index_global.index"], args)
    _print_or_status(plan, orch, show_status=args.show_status)
    if args.dry_run:
        return
    if not plan["build_list"] and not args.force:
        print("FAISS index already up-to-date")
        return
    _execute_plan(orch, plan, verbose=not args.quiet)


def cmd_eval(args: argparse.Namespace):
    config = _load_config(args)
    
    if not args.skip_build:
        orch = BuildOrchestrator(
            config=config,
            verbose=not args.quiet,
            inference_only=True,
            force_rebuild=args.force_build,
            dry_run=args.dry_run
        )
        result = orch.build_all(force=args.force_build, dry_run=args.dry_run)
        if not result.success:
            print("Auto-build failed:")
            for artifact, error in result.failed_artifacts.items():
                print(f"  - {artifact}: {error}")
            sys.exit(1)

    pipeline = NLPPipeline(config)
    pipeline.initialize()

    if args.data:
        data_path = _resolve(args.data)
    else:
        data_path = config.dataset.eval_csv

    if not data_path.exists():
        print(f"Eval CSV not found: {data_path}")
        sys.exit(1)
    
    _run_indomain_eval(pipeline, data_path, args)

def _run_indomain_eval(pipeline: NLPPipeline, data_path: Path, args):
    df = pd.read_csv(data_path)
    if not {"question", "tag"}.issubset(df.columns):
        print(f"CSV missing required columns at {data_path}")
        return

    total = len(df)
    top1 = 0
    topk = 0
    errors = 0
    misses = []
    results_data = []
    
    for idx, row in enumerate(df.itertuples(index=False)):
        try:
            result = pipeline.run(row.question, fusion_top_k=args.top_k)
            predicted = result.get("primary_tag")
            candidates = result.get("candidates", [])
            candidate_tags = [c.get("tag") for c in candidates[:args.top_k]]
            
            top_clf = candidates[0] if candidates else {}
            
            correct = predicted == row.tag
            if correct:
                top1 += 1
            else:
                misses.append({
                    "question": row.question,
                    "expected": row.tag,
                    "predicted": predicted,
                    "top_score": top_clf.get("final_score", 0.0) if top_clf else 0.0
                })
            
            if row.tag in candidate_tags:
                topk += 1
            
            results_data.append({
                "question": row.question,
                "expected_tag": row.tag,
                "predicted_tag": predicted,
                "correct": correct,
                "top_score": top_clf.get("final_score", 0.0) if top_clf else 0.0,
                "error": None
            })
        except Exception as e:
            errors += 1
            results_data.append({
                "question": row.question,
                "expected_tag": row.tag,
                "predicted_tag": None,
                "correct": False,
                "top_score": 0.0,
                "error": str(e)
            })
            if errors <= 5:
                logger.warning(f"Error on row {idx}: {e}")
        
        processed = idx + 1
        if processed % 100 == 0 or idx == total - 1:
            actual_acc = top1 / processed if processed > 0 else 0
            err_rate = errors / processed if processed > 0 else 0
            print(f"\rProcessed {processed}/{total} | Top-1: {actual_acc:.2%} | Errors: {err_rate:.1%}", end="", flush=True)
    
    print()

    if args.output:
        output_path = Path(args.output)
        pd.DataFrame(results_data).to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    processed_ok = total - errors
    print("\n" + "=" * 60)
    print("IN-DOMAIN EVALUATION")
    print("=" * 60)
    print(f"Total rows: {total} (processed: {processed_ok}, errors: {errors})")
    if total > 0:
        actual_top1 = top1 / total
        actual_topk = topk / total
        print(f"Top-1 accuracy (actual): {actual_top1:.2%} ({top1}/{total})")
        print(f"Top-{args.top_k} accuracy (actual): {actual_topk:.2%} ({topk}/{total})")
        if errors > 0 and processed_ok > 0:
            adj_top1 = top1 / processed_ok
            adj_topk = topk / processed_ok
            print(f"Top-1 accuracy (excl. errors): {adj_top1:.2%} ({top1}/{processed_ok})")
            print(f"Top-{args.top_k} accuracy (excl. errors): {adj_topk:.2%} ({topk}/{processed_ok})")
    else:
        print("Top-1 accuracy: N/A (no samples)")
    print(f"Top-1 misses: {len(misses)}")
    
    if misses:
        print(f"\nSample misses (up to 5):")
        for miss in misses[:5]:
            print(f"  - Expected: {miss['expected']}")
            print(f"    Got: {miss['predicted']}")
            print(f"    Question: {miss['question'][:50]}...")
    print("=" * 60)


def cmd_interactive(args: argparse.Namespace):
    config = _load_config(args)

    if not args.skip_build:
        orch = BuildOrchestrator(
            config=config,
            verbose=not args.quiet,
            inference_only=True,
            force_rebuild=args.force,
            dry_run=args.dry_run
        )
        result = orch.build_all(force=args.force, dry_run=args.dry_run)
        if not result.success:
            print("Build failed; cannot start interactive mode")
            sys.exit(1)

    pipeline = NLPPipeline(config)
    pipeline.initialize()

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            query = input("Query> ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            output = pipeline.run(query, fusion_top_k=args.top_k)
            answer = output.get("answer")
            candidates = output.get("candidates", [])
            print(f"Answer: {answer}")
            for cand in candidates[:3]:
                tag = cand.get("tag")
                score = cand.get("final_score", 0.0)
                print(f"  {tag}: {score:.4f}")
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"Error: {exc}")

def _get_cache_managers(config: NLPPipelineConfig):
    cache_base = config.output.cache_dir / "embeddings"
    return {
        "classifier": EmbeddingCacheManager(cache_base / "classifier", "classifier"),
        "sts": EmbeddingCacheManager(cache_base / "sts", "sts"),
    }


def cmd_cache_stats(args: argparse.Namespace):
    config = _load_config(args)
    managers = _get_cache_managers(config)
    
    total_size = 0
    total_entries = 0
    
    print("=" * 60)
    print("EMBEDDING CACHE STATISTICS")
    print("=" * 60)
    
    for cache_type, manager in managers.items():
        stats = manager.get_stats()
        total_size += stats.total_size_bytes
        total_entries += stats.total_entries
        
        print(f"\n{cache_type.upper()} CACHE:")
        print(f"  Location: {manager.cache_dir}")
        
        if stats.total_entries == 0:
            print("  Status: Empty")
            continue
        
        print(f"  Entries: {stats.total_entries:,}")
        print(f"  Tags: {stats.total_tags}")
        print(f"  Size: {stats.total_size_mb:.2f} MB")
        print(f"  Model: {stats.embedding_model}")
        print(f"  Dimension: {stats.embedding_dim}")
        
        if args.verbose and stats.entries_per_tag:
            print(f"\n  Entries per tag (top 10):")
            sorted_tags = sorted(stats.entries_per_tag.items(), key=lambda x: -x[1])
            for tag, count in sorted_tags[:10]:
                print(f"    {tag}: {count}")
            if len(sorted_tags) > 10:
                print(f"    ... and {len(sorted_tags) - 10} more tags")
    
    print("\n" + "-" * 60)
    print(f"TOTAL: {total_entries:,} entries, {total_size / (1024*1024):.2f} MB")
    print("=" * 60)
    
    if args.json:
        result = {
            cache_type: manager.get_stats().to_dict()
            for cache_type, manager in managers.items()
        }
        print("\nJSON output:")
        print(json.dumps(result, indent=2))


def cmd_cache_clear(args: argparse.Namespace):
    config = _load_config(args)
    managers = _get_cache_managers(config)
    
    targets = []
    if args.classifier:
        targets.append("classifier")
    if args.sts:
        targets.append("sts")
    if args.all or not targets:
        targets = list(managers.keys())
    
    if args.dry_run:
        print("DRY RUN - Would clear the following caches:")
        for target in targets:
            stats = managers[target].get_stats()
            print(f"  {target}: {stats.total_entries} entries, {stats.total_size_mb:.2f} MB")
        return
    
    for target in targets:
        manager = managers[target]
        stats = manager.get_stats()
        if stats.total_entries == 0:
            print(f"{target}: Already empty")
        else:
            manager.clear()
            print(f"{target}: Cleared {stats.total_entries} entries ({stats.total_size_mb:.2f} MB)")
    
    print("\nCache cleared successfully!")


def cmd_cache_gc(args: argparse.Namespace):
    config = _load_config(args)
    managers = _get_cache_managers(config)
    
    train_csv = config.dataset.train_csv
    if not train_csv.exists():
        print(f"Training data not found: {train_csv}")
        print("Cannot determine which embeddings are orphaned without training data.")
        sys.exit(1)
    
    print(f"Loading training data from {train_csv}...")
    df = pd.read_csv(train_csv)
    print(f"  {len(df)} questions loaded")
    
    current_fps = set()
    for _, row in df.iterrows():
        question = str(row['question']) if pd.notna(row['question']) else ""
        tag = str(row['tag']) if pd.notna(row['tag']) else ""
        fp = EmbeddingCacheManager.compute_fingerprint(question, tag)
        current_fps.add(fp)
    
    print(f"  {len(current_fps)} unique fingerprints")
    
    targets = []
    if args.classifier:
        targets.append("classifier")
    if args.sts:
        targets.append("sts")
    if args.all or not targets:
        targets = list(managers.keys())
    
    total_removed = 0
    total_freed = 0
    
    for target in targets:
        manager = managers[target]
        stats_before = manager.get_stats()
        
        if stats_before.total_entries == 0:
            print(f"\n{target}: Cache is empty, skipping")
            continue
        
        print(f"\n{target.upper()} CACHE:")
        print(f"  Before: {stats_before.total_entries} entries, {stats_before.total_size_mb:.2f} MB")
        
        if args.dry_run:
            changes = manager.detect_changes(current_fps)
            print(f"  Would remove: {len(changes.deleted)} orphaned entries")
        else:
            entries_removed, bytes_freed = manager.garbage_collect(current_fps)
            stats_after = manager.get_stats()
            
            total_removed += entries_removed
            total_freed += bytes_freed
            
            print(f"  Removed: {entries_removed} entries")
            print(f"  Freed: {bytes_freed / 1024:.1f} KB")
            print(f"  After: {stats_after.total_entries} entries, {stats_after.total_size_mb:.2f} MB")
    
    if not args.dry_run:
        print("\n" + "-" * 60)
        print(f"TOTAL: Removed {total_removed} entries, freed {total_freed / 1024:.1f} KB")
    
    print("\nGarbage collection complete!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pipelineNLP control CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    parser.add_argument("--config", type=Path, help="Path to JSON config file to override defaults")
    
    group = parser.add_argument_group("Path Configuration")
    group.add_argument("--train-csv", type=Path, help="Path to training CSV")
    group.add_argument("--eval-csv", type=Path, help="Path to eval CSV")
    group.add_argument("--output-dir", type=Path, help="Root directory for all build artifacts")

    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(p):
        p.add_argument("--force", action="store_true", help="Force rebuild even if fingerprints match")
        p.add_argument("--dry-run", action="store_true", help="Show actions without executing")
        p.add_argument("--quiet", action="store_true", help="Reduce logging noise")
        p.add_argument("--inference-only", action="store_true", help="Limit to inference artifacts")
        return p

    p_check = sub.add_parser("check", help="Show artifact status and pending rebuilds")
    add_common_flags(p_check)
    p_check.add_argument("--json", action="store_true", help="Emit status as JSON")
    p_check.set_defaults(func=cmd_check)

    p_build = sub.add_parser("build", help="Build all required artifacts")
    add_common_flags(p_build)
    p_build.add_argument("--show-status", action="store_true", help="Print status report before build")
    p_build.set_defaults(func=cmd_build)

    p_feat = sub.add_parser("features", help="Generate vocabulary and manual n-grams")
    add_common_flags(p_feat)
    p_feat.add_argument("--show-status", action="store_true")
    p_feat.set_defaults(func=cmd_features)

    p_clf = sub.add_parser("train-classifier", help="Train tag classifier artifacts")
    add_common_flags(p_clf)
    p_clf.add_argument("--show-status", action="store_true")
    p_clf.set_defaults(func=cmd_train_classifier)

    p_faiss = sub.add_parser("train-faiss", help="Build FAISS indices")
    add_common_flags(p_faiss)
    p_faiss.add_argument("--show-status", action="store_true")
    p_faiss.set_defaults(func=cmd_train_faiss)


    p_eval = sub.add_parser("eval", help="Evaluate pipeline accuracy")
    p_eval.add_argument("--data", help="CSV with question,tag (for indomain eval)")
    p_eval.add_argument("--top-k", type=int, default=1)
    p_eval.add_argument("--output", type=Path, help="Optional path to save detailed results CSV")
    p_eval.add_argument("--skip-build", action="store_true", help="Skip auto-build before eval")
    p_eval.add_argument("--force-build", action="store_true", help="Force build artifacts before eval")
    p_eval.add_argument("--quiet", action="store_true")
    p_eval.add_argument("--dry-run", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_int = sub.add_parser("interactive", help="Run interactive Q&A loop")
    p_int.add_argument("--top-k", type=int, default=3)
    p_int.add_argument("--skip-build", action="store_true")
    p_int.add_argument("--force", action="store_true")
    p_int.add_argument("--quiet", action="store_true")
    p_int.add_argument("--dry-run", action="store_true")
    p_int.set_defaults(func=cmd_interactive)

    p_cache_stats = sub.add_parser("cache-stats", help="Show embedding cache statistics")
    p_cache_stats.add_argument("--verbose", "-v", action="store_true", help="Show detailed per-tag breakdown")
    p_cache_stats.add_argument("--json", action="store_true", help="Output as JSON")
    p_cache_stats.set_defaults(func=cmd_cache_stats)

    p_cache_clear = sub.add_parser("cache-clear", help="Clear embedding cache")
    p_cache_clear.add_argument("--classifier", action="store_true", help="Clear only classifier cache")
    p_cache_clear.add_argument("--sts", action="store_true", help="Clear only STS cache")
    p_cache_clear.add_argument("--all", action="store_true", help="Clear all caches (default if none specified)")
    p_cache_clear.add_argument("--dry-run", action="store_true", help="Show what would be cleared")
    p_cache_clear.set_defaults(func=cmd_cache_clear)

    p_cache_gc = sub.add_parser("cache-gc", help="Garbage collect orphaned embeddings")
    p_cache_gc.add_argument("--classifier", action="store_true", help="GC only classifier cache")
    p_cache_gc.add_argument("--sts", action="store_true", help="GC only STS cache")
    p_cache_gc.add_argument("--all", action="store_true", help="GC all caches (default if none specified)")
    p_cache_gc.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    p_cache_gc.set_defaults(func=cmd_cache_gc)

    return parser

def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.debug)
    args.func(args)

if __name__ == "__main__":
    main()
