from __future__ import annotations

import argparse
import json
from loguru import logger
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .path_utils import (
    DATASETS_DIR,
    FEATURES_DIR,
    CLASSIFIER_MODELS_DIR as CLASSIFIER_DIR,
    SEMANTIC_MODELS_DIR as SEMANTIC_DIR,
)

def setup_logging(debug: bool = False):
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level
    )

class PreflightChecker:
    
    def __init__(
        self,
        datasets_dir: Optional[Path] = None,
        features_dir: Optional[Path] = None,
        classifier_dir: Optional[Path] = None,
        semantic_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
    ):
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DATASETS_DIR
        self.features_dir = Path(features_dir) if features_dir else FEATURES_DIR
        self.classifier_dir = Path(classifier_dir) if classifier_dir else CLASSIFIER_DIR
        self.semantic_dir = Path(semantic_dir) if semantic_dir else SEMANTIC_DIR
        self.embedding_model = embedding_model
        
        logger.debug(f"PreflightChecker initialized with:")
        logger.debug(f"  datasets_dir: {self.datasets_dir}")
        logger.debug(f"  features_dir: {self.features_dir}")
        logger.debug(f"  classifier_dir: {self.classifier_dir}")
        logger.debug(f"  semantic_dir: {self.semantic_dir}")
        logger.debug(f"  embedding_model: {self.embedding_model}")
    
    def check_datasets(self) -> Dict[str, str]:
        from nlpcomponents.build.fingerprint import compute_dataset_fingerprint
        
        logger.info("Checking datasets...")
        results = {}
        
        required = [
            ("question_tag.csv", ('question', 'tag')),
            ("eval.csv", ('question', 'tag'))
        ]
        
        for filename, columns in required:
            path = self.datasets_dir / filename
            try:
                if not path.exists():
                    results[filename] = "missing"
                    logger.warning(f"  {filename}: MISSING at {path}")
                else:
                    fp = compute_dataset_fingerprint(path, columns=columns)
                    if fp in ("missing", "unreadable", "no_matching_columns"):
                        results[filename] = fp
                        logger.warning(f"  {filename}: {fp}")
                    else:
                        results[filename] = "present"
                        logger.info(f"  {filename}: present (fingerprint: {fp[:12]}...)")
            except Exception as e:
                results[filename] = f"error: {str(e)}"
                logger.error(f"  {filename}: error checking - {e}")
        
        return results

    def check_and_deduplicate(
        self,
        auto_fix: bool = True
    ) -> Dict[str, Any]:
        logger.info("Checking for duplicate rows in question_tag.csv...")
        
        csv_path = self.datasets_dir / "question_tag.csv"
        result: Dict[str, Any] = {
            "status": "ok",
            "file": str(csv_path),
            "duplicates_found": 0,
            "duplicate_examples": [],
            "action_taken": None,
            "original_rows": 0,
            "final_rows": 0,
        }
        
        if not csv_path.exists():
            result["status"] = "file_missing"
            logger.warning(f"  question_tag.csv: MISSING at {csv_path}")
            return result
        
        try:
            df = pd.read_csv(csv_path)
            result["original_rows"] = len(df)
            
            if 'question' not in df.columns or 'tag' not in df.columns:
                result["status"] = "invalid_columns"
                logger.warning("  question_tag.csv: Missing required columns 'question' and/or 'tag'")
                return result
            
            duplicates_mask = df.duplicated(subset=['question', 'tag'], keep='first')
            num_duplicates = duplicates_mask.sum()
            
            result["duplicates_found"] = int(num_duplicates)
            
            if num_duplicates == 0:
                result["status"] = "ok"
                result["final_rows"] = len(df)
                logger.info(f"  question_tag.csv: No duplicates found ({len(df)} unique rows)")
                return result
            
            duplicate_rows = df[duplicates_mask].head(5)
            result["duplicate_examples"] = [
                {"question": row['question'][:50] + "..." if len(str(row['question'])) > 50 else row['question'], 
                 "tag": row['tag']}
                for _, row in duplicate_rows.iterrows()
            ]
            
            logger.warning(f"  question_tag.csv: Found {num_duplicates} duplicate (question, tag) pairs!")
            for i, example in enumerate(result["duplicate_examples"][:3]):
                logger.warning(f"    Example {i+1}: tag='{example['tag']}', question='{example['question']}'")
            
            if auto_fix:
                df_deduped = df.drop_duplicates(subset=['question', 'tag'], keep='first')
                result["final_rows"] = len(df_deduped)
                
                backup_path = csv_path.with_suffix('.csv.bak')
                df.to_csv(backup_path, index=False)
                logger.info(f"  Created backup at: {backup_path}")
                
                df_deduped.to_csv(csv_path, index=False)
                
                result["status"] = "fixed"
                result["action_taken"] = f"Removed {num_duplicates} duplicates, saved backup to {backup_path.name}"
                logger.info(f"  [FIXED] Removed {num_duplicates} duplicates ({result['original_rows']} -> {result['final_rows']} rows)")
                
                logger.warning("  [IMPORTANT] Embedding caches should be cleared after de-duplication!")
                logger.warning("  Run: python -m nlpcomponents.cli cache --clear")
            else:
                result["status"] = "duplicates_found"
                result["final_rows"] = result["original_rows"]
                result["action_taken"] = "No action (auto_fix=False)"
                logger.warning(f"  De-duplication skipped. Run with auto_fix=True to fix.")
            
            return result
            
        except Exception as e:
            result["status"] = f"error: {str(e)}"
            logger.error(f"  Error checking duplicates: {e}")
            return result

    def check_vocabulary(self) -> Dict[str, Any]:
        logger.info("Checking training vocabulary...")
        vocab_path = self.features_dir / "training_vocabulary.json"

        try:
            if not vocab_path.exists():
                logger.warning(f"  training_vocabulary.json: MISSING at {vocab_path}")
                return {"status": "missing", "path": str(vocab_path)}

            with vocab_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.warning("  training_vocabulary.json: invalid structure (not a JSON object)")
                return {"status": "invalid", "path": str(vocab_path)}

            has_unigrams = "word_unigrams" in data
            if not has_unigrams:
                logger.warning("  training_vocabulary.json: missing 'word_unigrams' key")
                return {"status": "invalid", "path": str(vocab_path)}

            stats = data.get("statistics", {})
            unigram_count = stats.get("total_word_unigrams")
            bigram_count = stats.get("total_word_bigrams")
            trigram_count = stats.get("total_word_trigrams")

            unigram_count = unigram_count or len(data.get("word_unigrams", []))
            bigram_count = bigram_count or len(data.get("word_bigrams", []))
            trigram_count = trigram_count or len(data.get("word_trigrams", []))

            logger.info(f"  training_vocabulary.json: present ({unigram_count} unigrams, {bigram_count} bigrams, {trigram_count} trigrams)")
            return {
                "status": "present",
                "path": str(vocab_path),
                "stats": {
                    "unigrams": unigram_count,
                    "bigrams": bigram_count,
                    "trigrams": trigram_count,
                },
            }
        except Exception as e:
            logger.error(f"  training_vocabulary.json: error checking - {e}")
            return {"status": f"error: {str(e)}", "path": str(vocab_path)}
    
    def check_ngrams(self) -> Dict[str, Any]:
        from nlpcomponents.build.fingerprint import compute_ngram_fingerprint
        
        logger.info("Checking n-gram features...")
        manual_path = self.features_dir / "manual_ngrams.json"
        
        try:
            if not manual_path.exists():
                logger.warning(f"  manual_ngrams.json: MISSING at {manual_path}")
                return {"status": "missing", "path": str(manual_path)}
            
            fp = compute_ngram_fingerprint(manual_path)
            if fp in ("missing", "unreadable", "invalid_json"):
                logger.warning(f"  manual_ngrams.json: {fp}")
                return {"status": fp, "path": str(manual_path)}
            
            logger.info(f"  manual_ngrams.json: present (fingerprint: {fp[:12]}...)")
            return {"status": "present", "fingerprint": fp}
        except Exception as e:
            logger.error(f"  manual_ngrams.json: error checking - {e}")
            return {"status": f"error: {str(e)}", "path": str(manual_path)}
    
    def check_classifier(self) -> Dict[str, Any]:
        from nlpcomponents.build.fingerprint import compute_classifier_fingerprint
        
        logger.info("Checking classifier...")
        model_path = self.classifier_dir / "unified_tag_classifier.pth"
        meta_path = self.classifier_dir / "unified_tag_classifier_metadata.json"
        
        try:
            if not model_path.exists():
                logger.warning(f"  classifier model: MISSING at {model_path}")
                return {"status": "missing", "path": str(model_path)}
            
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            if not meta_path.exists():
                logger.warning(f"  classifier metadata: MISSING at {meta_path}")
                return {"status": "no_metadata", "path": str(meta_path), "model_size_mb": model_size_mb}
            
            info = compute_classifier_fingerprint(meta_path)
            if not info['valid']:
                logger.warning(f"  classifier metadata: invalid")
                return {"status": "invalid_metadata", "path": str(meta_path), "model_size_mb": model_size_mb}
            
            fp_short = info['fingerprint'][:12] if info['fingerprint'] else 'None'
            logger.info(
                f"  classifier: present ({info.get('num_tags')} tags, "
                f"{model_size_mb:.1f} MB, fp: {fp_short}...)"
            )
            
            return {
                "status": "present",
                "fingerprint": info['fingerprint'],
                "num_tags": info.get('num_tags'),
                "best_val_acc": info.get('best_val_acc'),
                "model_size_mb": model_size_mb
            }
        except Exception as e:
            logger.error(f"  classifier: error checking - {e}")
            return {"status": f"error: {str(e)}", "path": str(model_path)}
    
    def check_faiss(self) -> Dict[str, Any]:
        from nlpcomponents.build.fingerprint import compute_faiss_fingerprint
        
        logger.info("Checking FAISS index...")
        index_path = self.semantic_dir / "faiss_index_global.index"
        embeddings_path = self.semantic_dir / "sts_embeddings.npy"
        mapping_path = self.semantic_dir / "question_mapping.csv"
        meta_path = self.semantic_dir / "sts_metadata.json"
        
        try:
            missing = []
            if not index_path.exists():
                missing.append("faiss_index_global.index")
            if not embeddings_path.exists():
                missing.append("sts_embeddings.npy")
            if not mapping_path.exists():
                missing.append("question_mapping.csv")
            
            if missing:
                logger.warning(f"  FAISS: MISSING files: {', '.join(missing)}")
                return {"status": "missing", "missing_files": missing}
            
            index_size_mb = index_path.stat().st_size / (1024 * 1024)
            embeddings_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
            
            if not meta_path.exists():
                logger.warning(f"  FAISS metadata: MISSING at {meta_path}")
                return {
                    "status": "no_metadata", 
                    "path": str(meta_path),
                    "index_size_mb": index_size_mb,
                    "embeddings_size_mb": embeddings_size_mb
                }
            
            info = compute_faiss_fingerprint(meta_path)
            if not info['valid']:
                logger.warning(f"  FAISS metadata: invalid")
                return {
                    "status": "invalid_metadata", 
                    "path": str(meta_path),
                    "index_size_mb": index_size_mb,
                    "embeddings_size_mb": embeddings_size_mb
                }
            
            fp_short = info['fingerprint'][:12] if info['fingerprint'] else 'None'
            
            use_unified = False
            embedding_dim = None
            try:
                with meta_path.open('r') as f:
                    meta_data = json.load(f)
                    use_unified = meta_data.get('use_unified_embeddings', False)
                    embedding_dim = meta_data.get('embedding_dim')
            except Exception:
                pass
            
            mode = "unified (256-dim)" if use_unified else f"raw ({embedding_dim or 1024}-dim)"
            logger.info(f"  FAISS: present ({info.get('num_questions')} questions, {mode}, "
                    f"index: {index_size_mb:.1f} MB, embeddings: {embeddings_size_mb:.1f} MB, fp: {fp_short}...)")
            
            return {
                "status": "present",
                "fingerprint": info['fingerprint'],
                "num_questions": info.get('num_questions'),
                "index_size_mb": index_size_mb,
                "embeddings_size_mb": embeddings_size_mb,
                "use_unified_embeddings": use_unified,
                "embedding_dim": embedding_dim
            }
        except Exception as e:
            logger.error(f"  FAISS: error checking - {e}")
            return {"status": f"error: {str(e)}"}
    
    def check_embedding_consistency(self, config_model: Optional[str] = None) -> Dict[str, Any]:
        from nlpcomponents.config import DEFAULT_EMBEDDING_MODEL
        
        logger.info("Checking embedding model consistency...")
        
        config_model = config_model or self.embedding_model or DEFAULT_EMBEDDING_MODEL
        results = {"config_model": config_model, "issues": []}
        

        clf_meta_path = self.classifier_dir / "unified_tag_classifier_metadata.json"
        if clf_meta_path.exists():
            try:
                with clf_meta_path.open('r', encoding='utf-8') as f:
                    clf_meta = json.load(f)
                clf_model = clf_meta.get("embedding_model")
                clf_dim = clf_meta.get("embedding_dim")
                
                results["classifier"] = {
                    "model": clf_model,
                    "dim": clf_dim
                }
                
                if clf_model and clf_model != config_model:
                    issue = f"Classifier trained with '{clf_model}' but config uses '{config_model}'"
                    results["issues"].append(issue)
                    logger.warning(f"  [MISMATCH] {issue}")
                else:
                    logger.info(f"  Classifier: {clf_model or 'unknown'} ({clf_dim or '?'}-dim)")
            except Exception as e:
                logger.warning(f"  Could not read classifier metadata: {e}")
        

        faiss_meta_path = self.semantic_dir / "sts_metadata.json"
        if faiss_meta_path.exists():
            try:
                with faiss_meta_path.open('r', encoding='utf-8') as f:
                    faiss_meta = json.load(f)
                faiss_model = faiss_meta.get("embedding_model")
                faiss_dim = faiss_meta.get("embedding_dim")
                
                results["faiss"] = {
                    "model": faiss_model,
                    "dim": faiss_dim
                }
                
                if faiss_model and faiss_model != config_model:
                    issue = f"FAISS trained with '{faiss_model}' but config uses '{config_model}'"
                    results["issues"].append(issue)
                    logger.warning(f"  [MISMATCH] {issue}")
                else:
                    logger.info(f"  FAISS: {faiss_model or 'unknown'} ({faiss_dim or '?'}-dim)")
            except Exception as e:
                logger.warning(f"  Could not read FAISS metadata: {e}")
        
        if results["issues"]:
            results["status"] = "mismatch"
            logger.warning(f"  Found {len(results['issues'])} embedding model mismatch(es)")
            logger.warning("  Run 'python -m nlpcomponents.cli build --force' to rebuild with current model")
        else:
            results["status"] = "consistent"
            logger.info("  All artifacts use consistent embedding model")
        
        return results
    
    def check_all(self, auto_deduplicate: bool = True) -> Dict[str, Dict]:
        logger.info("="*60)
        logger.info("PREFLIGHT CHECK - Validating all artifacts")
        logger.info("="*60)
        
        results = {}
        
        try:
            results["deduplication"] = self.check_and_deduplicate(auto_fix=auto_deduplicate)
        except Exception as e:
            logger.error(f"Deduplication check failed: {e}")
            results["deduplication"] = {"error": str(e)}
        
        try:
            results["datasets"] = self.check_datasets()
        except Exception as e:
            logger.error(f"Dataset check failed: {e}")
            results["datasets"] = {"error": str(e)}

        try:
            results["vocabulary"] = self.check_vocabulary()
        except Exception as e:
            logger.error(f"Vocabulary check failed: {e}")
            results["vocabulary"] = {"error": str(e)}
        
        try:
            results["ngrams"] = self.check_ngrams()
        except Exception as e:
            logger.error(f"N-gram check failed: {e}")
            results["ngrams"] = {"error": str(e)}
        
        try:
            results["classifier"] = self.check_classifier()
        except Exception as e:
            logger.error(f"Classifier check failed: {e}")
            results["classifier"] = {"error": str(e)}
        
        try:
            results["faiss"] = self.check_faiss()
        except Exception as e:
            logger.error(f"FAISS check failed: {e}")
            results["faiss"] = {"error": str(e)}
        
        try:
            from ..build.fingerprint import validate_artifact_consistency
            results["consistency"] = validate_artifact_consistency(
                datasets_dir=self.datasets_dir,
                features_dir=self.features_dir,
                classifier_dir=self.classifier_dir,
                semantic_dir=self.semantic_dir
            )
        except ImportError as e:
            logger.warning(f"Could not import consistency checker: {e}")
            results["consistency"] = {"status": "skipped", "reason": "import_error"}
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            results["consistency"] = {"error": str(e)}
        
        try:
            results["embedding_consistency"] = self.check_embedding_consistency(config_model=self.embedding_model)
        except Exception as e:
            logger.error(f"Embedding consistency check failed: {e}")
            results["embedding_consistency"] = {"error": str(e)}
        
        logger.info("="*60)
        return results
    
    def ensure_inference_ready(self, verbose: bool = True, auto_deduplicate: bool = True) -> bool:
        all_ready = True
        issues: List[str] = []
        
        try:
            dedup_result = self.check_and_deduplicate(auto_fix=auto_deduplicate)
            if dedup_result.get("status") == "duplicates_found":
                all_ready = False
                issues.append(f"Duplicates: {dedup_result.get('duplicates_found')} duplicate rows found")
            elif dedup_result.get("status") == "fixed":
                if verbose:
                    logger.info(f"  [FIXED] Removed {dedup_result.get('duplicates_found')} duplicate rows")
        except Exception as e:
            logger.warning(f"Deduplication check error: {e}")
        
        try:
            datasets = self.check_datasets()
            for name, status in datasets.items():
                if status != "present":
                    all_ready = False
                    issues.append(f"Dataset {name}: {status}")
        except Exception as e:
            all_ready = False
            issues.append(f"Dataset check error: {e}")

        try:
            vocab = self.check_vocabulary()
            if vocab.get("status") != "present":
                all_ready = False
                issues.append(f"Vocabulary: {vocab.get('status')}")
        except Exception as e:
            all_ready = False
            issues.append(f"Vocabulary check error: {e}")
        
        try:
            ngrams = self.check_ngrams()
            if ngrams.get("status") != "present":
                all_ready = False
                issues.append(f"N-grams: {ngrams.get('status')}")
        except Exception as e:
            all_ready = False
            issues.append(f"N-gram check error: {e}")
        
        try:
            classifier = self.check_classifier()
            if classifier.get("status") != "present":
                all_ready = False
                issues.append(f"Classifier: {classifier.get('status')}")
        except Exception as e:
            all_ready = False
            issues.append(f"Classifier check error: {e}")
        
        try:
            faiss = self.check_faiss()
            if faiss.get("status") != "present":
                all_ready = False
                issues.append(f"FAISS: {faiss.get('status')}")
        except Exception as e:
            all_ready = False
            issues.append(f"FAISS check error: {e}")
        
        try:
            embedding = self.check_embedding_consistency(config_model=self.embedding_model)
            if embedding.get("status") == "mismatch":
                all_ready = False
                for issue in embedding.get("issues", []):
                    issues.append(f"Embedding: {issue}")
        except Exception as e:
            logger.warning(f"Embedding consistency check error: {e}")
        
        if verbose:
            if all_ready:
                logger.info("[OK] All artifacts present and ready for inference")
            else:
                logger.warning("[FAIL] Artifacts not ready for inference:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
        
        return all_ready
    
    def get_rebuild_commands(self) -> List[str]:
        commands: List[str] = []
        
        try:
            status = self.check_all()
        except Exception as e:
            logger.error(f"Failed to check status for rebuild commands: {e}")
            commands.append(f"# Error checking status: {e}")
            return commands
        
        datasets = status.get("datasets", {})
        missing_datasets = [k for k, v in datasets.items() if v != "present"]
        if missing_datasets:
            commands.append(f"# Missing datasets: {', '.join(missing_datasets)}")
            commands.append("# Please provide the required CSV files in nlpcomponents/datasets/")
            return commands
        
        ngrams = status.get("ngrams", {})
        consistency = status.get("consistency", {})
        ngram_status = consistency.get("manual_ngrams.json", {}).get("status")

        vocab = status.get("vocabulary", {})
        if vocab.get("status") != "present":
            commands.append(f"# Vocabulary: {vocab.get('status')}")
            commands.append("python -m nlpcomponents.featurizer.build_vocabulary")

        if ngrams.get("status") != "present" or ngram_status == "stale":
            reason = ngrams.get("status") if ngrams.get("status") != "present" else "stale"
            commands.append(f"# N-grams: {reason}")
            commands.append(
                "python -m nlpcomponents.cli features"
            )
        
        classifier = status.get("classifier", {})
        clf_status = consistency.get("unified_tag_classifier.pth", {}).get("status")
        
        if classifier.get("status") != "present" or clf_status == "stale":
            reason = classifier.get("status") if classifier.get("status") != "present" else "stale"
            commands.append(f"# Classifier: {reason}")
            commands.append(
                "python -m nlpcomponents.cli train-classifier --force"
            )
        
        faiss = status.get("faiss", {})
        faiss_status = consistency.get("faiss_index_global.index", {}).get("status")
        
        if faiss.get("status") != "present" or faiss_status == "stale":
            reason = faiss.get("status") if faiss.get("status") != "present" else "stale"
            commands.append(f"# FAISS: {reason}")
            commands.append(
                "python -m nlpcomponents.cli train-faiss --force"
            )
        
        if not commands:
            commands.append("# All artifacts up-to-date")
        
        return commands

def preflight_check(verbose: bool = True) -> bool:
    try:
        checker = PreflightChecker()
        return checker.ensure_inference_ready(verbose=verbose)
    except Exception as e:
        logger.error(f"Preflight check failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Check pipelineNLP artifact status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for verbose output"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file to override defaults"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model to check for consistency"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for scripting)"
    )
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="Show commands needed to rebuild stale/missing artifacts"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code 1 if artifacts are not ready"
    )
    
    args = parser.parse_args()
    
    setup_logging(debug=args.debug)
    
    config = None
    if args.config:
        try:
            from nlpcomponents.config import NLPPipelineConfig
            with args.config.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
            config = NLPPipelineConfig.from_dict(config_data)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)

    datasets_dir = config.dataset.train_csv.parent if config else None
    features_dir = config.output.features_dir if config else None
    classifier_dir = config.output.classifier_dir if config else None
    semantic_dir = config.output.semantic_dir if config else None
    embedding_model = args.embedding_model or (config.semantic.embedding_model if config else None)

    checker = PreflightChecker(
        datasets_dir=datasets_dir,
        features_dir=features_dir,
        classifier_dir=classifier_dir,
        semantic_dir=semantic_dir,
        embedding_model=embedding_model,
    )
    
    if args.json:
        try:
            status = checker.check_all()
            is_ready = checker.ensure_inference_ready(verbose=False)
            output = {
                "ready": is_ready,
                "status": status,
                "rebuild_commands": checker.get_rebuild_commands() if not is_ready else []
            }
            print(json.dumps(output, indent=2, default=str))
            if args.strict and not is_ready:
                sys.exit(1)
        except Exception as e:
            error_output = {"error": str(e), "ready": False}
            print(json.dumps(error_output, indent=2))
            if args.strict:
                sys.exit(1)
    else:
        print("\n" + "="*60)
        print("PREFLIGHT CHECK - pipelineNLP Artifact Status")
        print("="*60)
        
        try:
            status = checker.check_all()
            
            print("\nCOMPONENT STATUS:")
            print("-"*40)
            
            datasets = status.get("datasets", {})
            print("\nDatasets:")
            for name, s in datasets.items():
                icon = "[OK]" if s == "present" else "[MISSING]"
                print(f"   {icon} {name}: {s}")

            vocab = status.get("vocabulary", {})
            print("\nTraining Vocabulary:")
            vocab_status = vocab.get("status", "unknown")
            icon = "[OK]" if vocab_status == "present" else "[MISSING]"
            print(f"   {icon} training_vocabulary.json: {vocab_status}")
            stats = vocab.get("stats", {}) if isinstance(vocab, dict) else {}
            if vocab_status == "present" and stats:
                print(f"      unigrams: {stats.get('unigrams')}")
                print(f"      bigrams: {stats.get('bigrams')}")
                print(f"      trigrams: {stats.get('trigrams')}")
            
            ngrams = status.get("ngrams", {})
            print("\nN-gram Features:")
            ngram_status = ngrams.get("status", "unknown")
            icon = "[OK]" if ngram_status == "present" else "[MISSING]"
            print(f"   {icon} manual_ngrams.json: {ngram_status}")
            if ngrams.get("fingerprint"):
                print(f"      fingerprint: {ngrams['fingerprint'][:16]}...")
            
            classifier = status.get("classifier", {})
            print("\nClassifier:")
            clf_status = classifier.get("status", "unknown")
            icon = "[OK]" if clf_status == "present" else "[MISSING]"
            print(f"   {icon} unified_tag_classifier.pth: {clf_status}")
            if clf_status == "present":
                print(f"      tags: {classifier.get('num_tags')}")
                print(f"      val_acc: {(classifier.get('best_val_acc') or 0):.2f}%")
                print(f"      size: {(classifier.get('model_size_mb') or 0):.1f} MB")
            
            faiss = status.get("faiss", {})
            print("\nFAISS Index:")
            faiss_status = faiss.get("status", "unknown")
            icon = "[OK]" if faiss_status == "present" else "[MISSING]"
            print(f"   {icon} faiss_index_global.index: {faiss_status}")
            if faiss_status == "present":
                print(f"      questions: {faiss.get('num_questions')}")
                print(f"      index size: {(faiss.get('index_size_mb') or 0):.1f} MB")
                print(f"      embeddings size: {(faiss.get('embeddings_size_mb') or 0):.1f} MB")
                use_unified = faiss.get('use_unified_embeddings', False)
                print(f"      unified embeddings: {'yes' if use_unified else 'no (legacy)'}")
            
            consistency = status.get("consistency", {})
            print("\nConsistency Check:")
            for artifact, info in consistency.items():
                if isinstance(info, dict):
                    art_status = info.get("status", "unknown")
                    icon = "[OK]" if art_status in ("up-to-date", "present") else "[STALE]" if art_status == "stale" else "[MISSING]"
                    print(f"   {icon} {artifact}: {art_status}")
                    if info.get("reason"):
                        print(f"      reason: {info['reason']}")
            
            is_ready = checker.ensure_inference_ready(verbose=False)
            print("\n" + "="*60)
            if is_ready:
                print("[SUCCESS] ALL ARTIFACTS READY FOR INFERENCE")
            else:
                print("[FAIL] ARTIFACTS NOT READY")
            print("="*60)
            
            if args.show_commands or not is_ready:
                commands = checker.get_rebuild_commands()
                if commands and commands[0] != "# All artifacts up-to-date":
                    print("\nREBUILD OPTIONS:")
                    print("-"*40)
                    print("   # Recommended: Use the build orchestrator")
                    print("   python -m nlpcomponents.cli build")
                    print()
                    print("   # Or run individual commands:")
                    for cmd in commands:
                        print(f"   {cmd}")
                    print()
            
            if args.strict and not is_ready:
                sys.exit(1)
                
        except Exception as e:
            print(f"\n[ERROR] {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            if args.strict:
                sys.exit(1)

if __name__ == "__main__":
    main()
