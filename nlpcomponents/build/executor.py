
from __future__ import annotations

from loguru import logger
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from ..config import NLPPipelineConfig
from ..utils.path_utils import PROJECT_ROOT

@dataclass
class BuildStepResult:
    success: bool
    error: Optional[str] = None
    duration: Optional[float] = None
    output: Optional[str] = None

class BuildExecutor:

    def __init__(
        self,
        config: NLPPipelineConfig,
        verbose: bool = True
    ):
        self.config = config
        self.features_dir = config.output.features_dir
        self.classifier_dir = config.output.classifier_dir
        self.semantic_dir = config.output.semantic_dir
        self.verbose = verbose

        logger.debug(f"BuildExecutor initialized")
        logger.debug(f"  features_dir: {self.features_dir}")
        logger.debug(f"  classifier_dir: {self.classifier_dir}")
        logger.debug(f"  semantic_dir: {self.semantic_dir}")

    def _prefix_args(self) -> List[str]:
        prefixes = self.config.prefixes
        args = [
            "--use-native-prompts" if prefixes.use_native_prompts else "--no-use-native-prompts",
            "--use-prefixes" if prefixes.use_prefixes else "--no-use-prefixes",
            "--use-instruct-format" if prefixes.use_instruct_format else "--no-use-instruct-format",
            "--instruct-task", prefixes.instruct_task,
            "--sts-query-prefix", prefixes.sts_query_prefix,
            "--sts-passage-prefix", prefixes.sts_passage_prefix,
            "--classifier-query-prefix", prefixes.classifier_query_prefix,
        ]
        return args

    def _get_build_command(self, artifact_name: str) -> List[str]:
        if artifact_name == "training_vocabulary.json":
            cmd = [
                "nlpcomponents/featurizer/build_vocabulary.py",
                "--train-csv", str(self.config.dataset.train_csv),
                "--output", str(self.config.output.features_dir / "training_vocabulary.json")
            ]
            return cmd
            
        elif artifact_name == "manual_ngrams.json":
            return [
                "nlpcomponents/featurizer/generate_features.py",
                "--train-csv", str(self.config.dataset.train_csv),
                "--output-dir", str(self.config.output.features_dir),
                "--top-k", "40",
                "--auto-clean",
                "--use-tfidf",
                "--force-overwrite"
            ]
            
        elif artifact_name == "unified_tag_classifier.pth":
            return [
                "nlpcomponents/training/train_tag_classifier.py",
                "--train-csv", str(self.config.dataset.train_csv),
                "--eval-csv", str(self.config.dataset.eval_csv),
                "--features-dir", str(self.config.output.features_dir),
                "--models", str(self.config.output.classifier_dir),
                "--embedding-model", self.config.classifier.embedding_model,
                "--force",
            ] + self._prefix_args()
            
        elif artifact_name == "faiss_index_global.index":
            cmd = [
                "nlpcomponents/training/build_faiss_indices.py",
                "--train-csv", str(self.config.dataset.train_csv),
                "--models-dir", str(self.config.output.semantic_dir),
                "--classifier-dir", str(self.config.output.classifier_dir),
                "--embedding-model", self.config.semantic.embedding_model,
                "--global",
                "--force",
            ] + self._prefix_args()
            if not self.config.semantic.normalize_embeddings:
                cmd.append("--no-normalize")
            return cmd

        raise ValueError(f"Unknown artifact: {artifact_name}")

    def _get_artifact_paths(self, artifact_name: str) -> List[Path]:
        paths = []
        
        if artifact_name == "training_vocabulary.json":
            paths.append(self.config.output.features_dir / "training_vocabulary.json")

        elif artifact_name == "manual_ngrams.json":
            paths.append(self.config.output.features_dir / "manual_ngrams.json")
            paths.append(self.config.output.features_dir / "auto_ngrams.json")
            paths.append(self.config.output.features_dir / "overlap_analysis.json")
            paths.append(self.config.output.features_dir / "cleanup_report.json")

        elif artifact_name == "unified_tag_classifier.pth":
            paths.append(self.config.output.classifier_dir / "unified_tag_classifier.pth")
            paths.append(self.config.output.classifier_dir / "unified_tag_classifier_metadata.json")

        elif artifact_name == "faiss_index_global.index":
            paths.append(self.config.output.semantic_dir / "faiss_index_global.index")
            paths.append(self.config.output.semantic_dir / "sts_embeddings.npy")
            paths.append(self.config.output.semantic_dir / "question_mapping.csv")
            paths.append(self.config.output.semantic_dir / "sts_metadata.json")

        return [p for p in paths if p.exists()]

    def _validate_build_output(self, artifact_name: str) -> bool:
        if artifact_name == "training_vocabulary.json":
            vocab_path = self.features_dir / "training_vocabulary.json"
            if not vocab_path.exists():
                logger.error("  Validation failed: missing training_vocabulary.json")
                return False
            try:
                import json
                with vocab_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    logger.error("  Validation failed: vocabulary file is not a JSON object")
                    return False

                has_unigrams = "word_unigrams" in data
                if not has_unigrams:
                    logger.error("  Validation failed: missing 'word_unigrams' key in vocabulary")
                    return False
            except Exception as e:
                logger.error(f"  Validation failed: {e}")
                return False

            logger.debug(f"  Validation passed for {artifact_name}")
            return True

        if artifact_name == "manual_ngrams.json":
            manual_path = self.features_dir / "manual_ngrams.json"
            auto_path = self.features_dir / "auto_ngrams.json"

            if not manual_path.exists() or not auto_path.exists():
                logger.error(f"  Validation failed: missing n-gram files")
                return False

            try:
                import json
                with manual_path.open('r') as f:
                    data = json.load(f)
                if 'metadata' not in data or 'tags' not in data:
                    logger.error(f"  Validation failed: invalid n-gram JSON structure (expected 'metadata' and 'tags' keys)")
                    return False
            except Exception as e:
                logger.error(f"  Validation failed: {e}")
                return False

            logger.debug(f"  Validation passed for {artifact_name}")
            return True

        elif artifact_name == "unified_tag_classifier.pth":
            model_path = self.classifier_dir / "unified_tag_classifier.pth"
            metadata_path = self.classifier_dir / "unified_tag_classifier_metadata.json"

            if not model_path.exists():
                logger.error(f"  Validation failed: missing .pth file")
                return False

            if not metadata_path.exists():
                logger.error(f"  Validation failed: missing metadata.json")
                return False

            try:
                import json
                with metadata_path.open('r') as f:
                    metadata = json.load(f)
                if 'fingerprint' not in metadata or 'num_tags' not in metadata:
                    logger.error(f"  Validation failed: invalid metadata structure")
                    return False
            except Exception as e:
                logger.error(f"  Validation failed: {e}")
                return False

            logger.debug(f"  Validation passed for {artifact_name}")
            return True

        elif artifact_name == "faiss_index_global.index":
            index_path = self.semantic_dir / "faiss_index_global.index"
            embeddings_path = self.semantic_dir / "sts_embeddings.npy"
            mapping_path = self.semantic_dir / "question_mapping.csv"
            metadata_path = self.semantic_dir / "sts_metadata.json"

            missing = []
            if not index_path.exists():
                missing.append("faiss_index_global.index")
            if not embeddings_path.exists():
                missing.append("sts_embeddings.npy")
            if not mapping_path.exists():
                missing.append("question_mapping.csv")
            if not metadata_path.exists():
                missing.append("sts_metadata.json")

            if missing:
                logger.error(f"  Validation failed: missing files: {', '.join(missing)}")
                return False

            try:
                import json
                with metadata_path.open('r') as f:
                    metadata = json.load(f)
                if 'fingerprint' not in metadata or 'num_questions' not in metadata:
                    logger.error(f"  Validation failed: invalid metadata structure")
                    return False
            except Exception as e:
                logger.error(f"  Validation failed: {e}")
                return False

            logger.debug(f"  Validation passed for {artifact_name}")
            return True

        else:
            logger.warning(f"  No validation logic for {artifact_name}, assuming OK")
            return True

    def execute_build(
        self,
        artifact_name: str,
        step: int,
        total: int
    ) -> BuildStepResult:
        start_time = time.time()

        try:
            cmd_args = self._get_build_command(artifact_name)
        except ValueError as e:
            return BuildStepResult(success=False, error=str(e))

        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.classifier_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_dir.mkdir(parents=True, exist_ok=True)

        full_cmd = [sys.executable] + cmd_args
        logger.info(f"  Running: {' '.join(full_cmd)}")

        try:
            result = subprocess.run(
                full_cmd,
                cwd=PROJECT_ROOT,
                env=os.environ,
                check=False,
                capture_output=not self.verbose,
                text=True,
                timeout=3600
            )

            if result.returncode != 0:
                error_msg = f"Build command failed with exit code {result.returncode}"
                if result.stderr:
                    truncated = result.stderr[:500] + ("..." if len(result.stderr) > 500 else "")
                    error_msg += f": {truncated}"
                    if len(result.stderr) > 500:
                        logger.debug(f"Full stderr for {artifact_name}: {result.stderr}")

                logger.error(f"  {error_msg}")

                return BuildStepResult(
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time,
                    output=result.stderr if result.stderr else result.stdout
                )

            if not self._validate_build_output(artifact_name):
                logger.error(f"  Build validation failed")

                return BuildStepResult(
                    success=False,
                    error="Output validation failed",
                    duration=time.time() - start_time
                )

            duration = time.time() - start_time
            logger.info(f"  Build completed in {duration:.1f}s")

            return BuildStepResult(
                success=True,
                duration=duration,
                output=result.stdout if result.stdout else None
            )

        except subprocess.TimeoutExpired:
            error_msg = "Build timed out after 1 hour"
            logger.error(f"  {error_msg}")

            return BuildStepResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Build failed with exception: {e}"
            logger.error(f"  {error_msg}")

            return BuildStepResult(
                success=False,
                error=error_msg,
                duration=time.time() - start_time
            )
