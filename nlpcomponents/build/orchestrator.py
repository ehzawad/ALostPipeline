
from __future__ import annotations

import json
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from contextlib import contextmanager

from filelock import FileLock, Timeout as FileLockTimeout

from ..config import NLPPipelineConfig

@contextmanager
def build_lock(lock_file: Path, timeout: float = 300.0):
    lock_file = Path(lock_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(lock_file), timeout=timeout)
    try:
        lock.acquire()
        logger.debug(f"Acquired build lock: {lock_file}")
        yield
    except FileLockTimeout:
        raise TimeoutError(
            f"Could not acquire build lock after {timeout}s. "
            f"Another process may be building. Lock file: {lock_file}"
        )
    finally:
        try:
            lock.release()
            logger.debug(f"Released build lock: {lock_file}")
        except Exception as e:
            logger.warning(f"Error releasing lock: {e}")
        try:
            if lock_file.exists():
                lock_file.unlink()
        except OSError:
            pass

class ArtifactStatus(Enum):
    MISSING = "missing"
    PARTIAL = "partial"
    STALE = "stale"
    UP_TO_DATE = "up_to_date"
    ERROR = "error"

@dataclass
class BuildResult:
    success: bool
    rebuilt_artifacts: List[str] = field(default_factory=list)
    skipped_artifacts: List[str] = field(default_factory=list)
    failed_artifacts: Dict[str, str] = field(default_factory=dict)
    duration: Optional[float] = None

class BuildOrchestrator:

    INFERENCE_ARTIFACTS = [
        "training_vocabulary.json",
        "manual_ngrams.json",
        "unified_tag_classifier.pth",
        "faiss_index_global.index",
    ]

    def __init__(
        self,
        config: Optional[NLPPipelineConfig] = None,
        verbose: bool = True,
        inference_only: bool = False,
        force_rebuild: bool = False,
        dry_run: bool = False
    ):
        self.config = config or NLPPipelineConfig()
        self.verbose = verbose
        self.inference_only = inference_only
        self.force_rebuild = force_rebuild
        self.dry_run = dry_run
        
        self.dependency_graph = self._build_dependency_graph()

        logger.info("Initializing build orchestrator...")
        if self.force_rebuild:
            logger.warning("  Force rebuild enabled (will ignore content fingerprints)")
        if self.dry_run:
            logger.warning("  Dry run enabled (no changes will be made)")

        from .fingerprint import (
            compute_dataset_fingerprint,
            compute_ngram_fingerprint,
        )
        self._compute_dataset_fingerprint = compute_dataset_fingerprint
        self._compute_ngram_fingerprint = compute_ngram_fingerprint

        logger.debug(f"BuildOrchestrator initialized with config: {self.config.dataset}")

    def _build_dependency_graph(self) -> Dict[str, Dict]:
        return {
            "training_vocabulary.json": {
                "type": "feature",
                "path": self.config.output.features_dir / "training_vocabulary.json",
                "dependencies": ["question_tag.csv"],
                "dependents": [],
                "companion_files": []
            },
            "manual_ngrams.json": {
                "type": "feature",
                "path": self.config.output.features_dir / "manual_ngrams.json",
                "dependencies": ["question_tag.csv"],
                "dependents": ["unified_tag_classifier.pth"],
                "companion_files": []
            },
            "unified_tag_classifier.pth": {
                "type": "model",
                "path": self.config.output.classifier_dir / "unified_tag_classifier.pth",
                "dependencies": ["question_tag.csv", "eval.csv", "manual_ngrams.json"],
                "dependents": [],
                "companion_files": ["unified_tag_classifier_metadata.json"]
            },
            "faiss_index_global.index": {
                "type": "model",
                "path": self.config.output.semantic_dir / "faiss_index_global.index",
                "dependencies": ["question_tag.csv"],
                "dependents": [],
                "companion_files": ["sts_metadata.json", "sts_embeddings.npy", "question_mapping.csv"]
            }
        }

    def _get_current_dataset_fingerprints(self) -> Dict[str, str]:
        current_fps = {
            "question_tag.csv": self._compute_dataset_fingerprint(
                self.config.dataset.train_csv, columns=('question', 'tag')
            ),
            "eval.csv": self._compute_dataset_fingerprint(
                self.config.dataset.eval_csv, columns=('question', 'tag')
            )
        }

        logger.debug(f"Current dataset fingerprints:")
        for name, fp in current_fps.items():
            logger.debug(f"  {name}: {fp[:12]}...")

        return current_fps

    def _validate_artifact(self, artifact_name: str) -> ArtifactStatus:
        info = self.dependency_graph.get(artifact_name)
        if not info:
            logger.error(f"Unknown artifact: {artifact_name}")
            return ArtifactStatus.ERROR
            
        path = info["path"]
        
        if artifact_name in ["training_vocabulary.json", "manual_ngrams.json"]:
            if not path.exists():
                logger.debug(f"  {artifact_name}: MISSING (file not found at {path})")
                return ArtifactStatus.MISSING
            try:
                with path.open('r', encoding='utf-8') as f:
                    json.load(f)
                logger.debug(f"  {artifact_name}: present and valid")
                return ArtifactStatus.UP_TO_DATE
            except Exception as e:
                logger.warning(f"  {artifact_name}: corrupt JSON ({e}), treating as MISSING")
                return ArtifactStatus.MISSING
                
        elif artifact_name == "unified_tag_classifier.pth":
            metadata_path = self.config.output.classifier_dir / "unified_tag_classifier_metadata.json"

            if metadata_path.exists() and not path.exists():
                logger.debug(f"  {artifact_name}: PARTIAL (metadata exists but .pth missing)")
                return ArtifactStatus.PARTIAL

            if not path.exists():
                logger.debug(f"  {artifact_name}: MISSING (file not found)")
                return ArtifactStatus.MISSING

            logger.debug(f"  {artifact_name}: present")
            return ArtifactStatus.UP_TO_DATE
            
        elif artifact_name == "faiss_index_global.index":
            embeddings_path = self.config.output.semantic_dir / "sts_embeddings.npy"
            metadata_path = self.config.output.semantic_dir / "sts_metadata.json"
            mapping_path = self.config.output.semantic_dir / "question_mapping.csv"

            has_metadata = metadata_path.exists()
            has_mapping = mapping_path.exists()
            has_index = path.exists()
            has_embeddings = embeddings_path.exists()

            if not (has_index and has_embeddings and has_mapping and has_metadata):
                missing_parts = []
                if not has_index:
                    missing_parts.append("index")
                if not has_embeddings:
                    missing_parts.append("embeddings")
                if not has_mapping:
                    missing_parts.append("mapping")
                if not has_metadata:
                    missing_parts.append("metadata")

                logger.debug(f"  {artifact_name}: MISSING ({', '.join(missing_parts)} not found)")
                return ArtifactStatus.MISSING

            logger.debug(f"  {artifact_name}: present")
            return ArtifactStatus.UP_TO_DATE
            
        return ArtifactStatus.ERROR

    def _dependencies_changed(self, artifact_name: str, current_fps: Dict[str, str]) -> bool:
        info = self.dependency_graph.get(artifact_name)
        if info is None:
            logger.warning(f"Unknown artifact in _dependencies_changed: {artifact_name}")
            return True
        path = info["path"]
        
        if artifact_name in ["training_vocabulary.json", "manual_ngrams.json"]:
            if not path.exists():
                return True
            try:
                with path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                metadata = data.get('metadata', {})
                deps = metadata.get('dependencies', {})
                stored_fp = deps.get('dataset', {}).get('fingerprint')
                
                if not stored_fp:
                    return True
                    
                if stored_fp != current_fps["question_tag.csv"]:
                    logger.debug(f"  {artifact_name}: dataset changed")
                    return True
                return False
            except Exception as e:
                logger.debug(f"  {artifact_name}: error checking dependencies, assuming changed: {e}")
                return True

        elif artifact_name == "unified_tag_classifier.pth":
            metadata_file = self.config.output.classifier_dir / "unified_tag_classifier_metadata.json"
            if not metadata_file.exists():
                return True
            try:
                with metadata_file.open('r', encoding='utf-8') as f:
                    metadata = json.load(f)
                deps = metadata.get("dependencies", {})

                # Check training dataset fingerprint
                stored_dataset_fp = deps.get("dataset", {}).get("fingerprint")
                if stored_dataset_fp != current_fps["question_tag.csv"]:
                    logger.debug(f"  {artifact_name}: training dataset changed")
                    return True

                # Check eval dataset fingerprint (Bug 7 fix: catch None mismatches)
                stored_eval_fp = deps.get("eval_dataset", {}).get("fingerprint")
                current_eval_fp = current_fps.get("eval.csv")
                if stored_eval_fp != current_eval_fp:
                    if stored_eval_fp is None or current_eval_fp is None:
                        logger.debug(f"  {artifact_name}: eval dataset fingerprint missing (stored={stored_eval_fp is not None}, current={current_eval_fp is not None})")
                    else:
                        logger.debug(f"  {artifact_name}: eval dataset changed")
                    return True

                # Check ngram fingerprint
                current_ngram_fp = self._compute_ngram_fingerprint(
                    self.config.output.features_dir / "manual_ngrams.json"
                )
                stored_ngram_fp = deps.get("ngrams", {}).get("fingerprint")
                if stored_ngram_fp != current_ngram_fp:
                    logger.debug(f"  {artifact_name}: n-grams changed")
                    return True

                # Check embedding model (Bug 3 fix: track hyperparameters)
                stored_embedding_model = metadata.get("embedding_model")
                if stored_embedding_model != self.config.classifier.embedding_model:
                    logger.debug(f"  {artifact_name}: embedding model changed ({stored_embedding_model} -> {self.config.classifier.embedding_model})")
                    return True

                # Check normalization setting
                stored_normalize = metadata.get("normalize_embeddings")
                if stored_normalize is not None and stored_normalize != self.config.classifier.normalize_embeddings:
                    logger.debug(f"  {artifact_name}: normalization setting changed")
                    return True

                # Check prefix config
                from ..cache.embedding_cache import get_prefix_config_hash
                current_prefix_hash = get_prefix_config_hash(self.config.prefixes)
                stored_prefix_hash = metadata.get("prefix_config_hash")
                if stored_prefix_hash is not None and stored_prefix_hash != current_prefix_hash:
                    logger.debug(f"  {artifact_name}: prefix config changed")
                    return True

                return False
            except Exception as e:
                logger.debug(f"  {artifact_name}: error checking dependencies, assuming changed: {e}")
                return True

        elif artifact_name == "faiss_index_global.index":
            metadata_file = self.config.output.semantic_dir / "sts_metadata.json"
            if not metadata_file.exists():
                return True
            try:
                with metadata_file.open('r', encoding='utf-8') as f:
                    metadata = json.load(f)
                deps = metadata.get("dependencies", {})

                # Check training dataset fingerprint
                stored_dataset_fp = deps.get("dataset", {}).get("fingerprint")
                if stored_dataset_fp != current_fps["question_tag.csv"]:
                    logger.debug(f"  {artifact_name}: training dataset changed")
                    return True

                # Check embedding model (Bug 3 fix: track hyperparameters)
                stored_embedding_model = metadata.get("embedding_model")
                if stored_embedding_model != self.config.semantic.embedding_model:
                    logger.debug(f"  {artifact_name}: embedding model changed ({stored_embedding_model} -> {self.config.semantic.embedding_model})")
                    return True

                # Check normalization setting
                stored_normalize = metadata.get("normalize_embeddings")
                if stored_normalize is not None and stored_normalize != self.config.semantic.normalize_embeddings:
                    logger.debug(f"  {artifact_name}: normalization setting changed")
                    return True

                # Check prefix config for STS
                from ..cache.embedding_cache import get_prefix_config_hash
                current_prefix_hash = get_prefix_config_hash(self.config.prefixes)
                # STS metadata stores individual prefix fields, not hash - check those
                stored_use_native = metadata.get("use_native_prompts")
                stored_use_prefixes = metadata.get("use_prefixes")
                stored_use_instruct = metadata.get("use_instruct_format")
                if stored_use_native is not None and stored_use_native != self.config.prefixes.use_native_prompts:
                    logger.debug(f"  {artifact_name}: use_native_prompts changed")
                    return True
                if stored_use_prefixes is not None and stored_use_prefixes != self.config.prefixes.use_prefixes:
                    logger.debug(f"  {artifact_name}: use_prefixes changed")
                    return True
                if stored_use_instruct is not None and stored_use_instruct != self.config.prefixes.use_instruct_format:
                    logger.debug(f"  {artifact_name}: use_instruct_format changed")
                    return True

                return False
            except Exception as e:
                logger.debug(f"  {artifact_name}: error checking dependencies, assuming changed: {e}")
                return True
                
        return True

    def _topological_sort(self, artifacts: Set[str]) -> List[str]:
        order = []
        if "training_vocabulary.json" in artifacts:
            order.append("training_vocabulary.json")
        if "manual_ngrams.json" in artifacts:
            order.append("manual_ngrams.json")
        if "unified_tag_classifier.pth" in artifacts:
            order.append("unified_tag_classifier.pth")
        if "faiss_index_global.index" in artifacts:
            order.append("faiss_index_global.index")
        return order

    def calculate_rebuild_set(
        self,
        force: bool = False,
        artifacts: Optional[List[str]] = None
    ) -> List[str]:
        logger.info("Calculating rebuild set...")
        rebuild_needed = set()

        current_fps = self._get_current_dataset_fingerprints()

        all_artifacts = list(self.dependency_graph.keys())
        if artifacts:
            all_artifacts = [a for a in all_artifacts if a in artifacts]

        if self.inference_only:
            all_artifacts = [a for a in all_artifacts if a in self.INFERENCE_ARTIFACTS]
            logger.debug(f"Inference-only mode: filtering to {len(all_artifacts)} artifacts")

        for artifact_name in all_artifacts:
            if force:
                rebuild_needed.add(artifact_name)
                continue
                 
            status = self._validate_artifact(artifact_name)
            if status in (ArtifactStatus.MISSING, ArtifactStatus.PARTIAL):
                rebuild_needed.add(artifact_name)
                continue
            
            if self._dependencies_changed(artifact_name, current_fps):
                rebuild_needed.add(artifact_name)
        
        changed = True
        while changed:
            changed = False
            for artifact_name, info in self.dependency_graph.items():
                if artifact_name in rebuild_needed or artifact_name not in all_artifacts:
                    continue
                deps = info.get("dependencies", [])
                for dep in deps:
                    if dep in rebuild_needed and dep in self.dependency_graph:
                        logger.info(f"  {artifact_name}: REBUILD (dependency {dep} rebuilding)")
                        rebuild_needed.add(artifact_name)
                        changed = True
                        break

        rebuild_list = self._topological_sort(rebuild_needed)
        return rebuild_list

    def create_build_plan(self) -> Dict[str, Any]:
        rebuild_list = self.calculate_rebuild_set(force=self.force_rebuild)
        status = self.analyze_status()
        return {
            "build_list": rebuild_list,
            "status": status
        }

    def print_status_report(self, plan: Dict[str, Any]) -> None:
        status = plan["status"]
        build_list = set(plan["build_list"])
        
        lines = []
        lines.append("=" * 60)
        lines.append("BUILD ORCHESTRATOR - Status Report")
        lines.append("=" * 60)

        lines.append("\nDATASETS:")
        for name in ["question_tag.csv", "eval.csv"]:
            info = status.get(name, {})
            s = info.get("status", "unknown")
            fp = info.get("fingerprint", "")
            icon = "[OK]" if s == "present" else "[MISSING]"
            fp_str = f" (fp: {fp[:12]}...)" if fp else ""
            lines.append(f"  {icon} {name}{fp_str}")

        lines.append("\nARTIFACTS:")
        for artifact_name in self.dependency_graph.keys():
            info = status.get(artifact_name, {})
            s = info.get("status", "unknown")
            reason = info.get("reason")

            action = "SKIP"
            needs_rebuild = artifact_name in build_list
            if needs_rebuild:
                action = "REBUILD"

            if s == "up_to_date" and needs_rebuild:
                icon = "[CASCADE]"
                reason = "dependency being rebuilt"
            elif s == "up_to_date":
                icon = "[UP-TO-DATE]"
            elif s == "stale":
                icon = "[STALE]"
            elif s == "partial":
                icon = "[PARTIAL]"
            elif s == "missing":
                icon = "[MISSING]"
            else:
                icon = "[ERROR]"

            lines.append(f"  {icon} {artifact_name}")
            if reason:
                lines.append(f"      Reason: {reason}")
            lines.append(f"      Action: {action}")

        lines.append("\n" + "=" * 60)
        print("\n".join(lines))

    def execute_plan(self, plan: Dict[str, Any], executor: Any) -> bool:
        rebuild_list = plan["build_list"]
        
        if not rebuild_list:
            logger.info("All artifacts up-to-date, nothing to build")
            return True

        if self.dry_run:
            logger.info(f"DRY RUN: Would rebuild {len(rebuild_list)} artifacts:")
            for i, artifact in enumerate(rebuild_list, 1):
                logger.info(f"  {i}. {artifact}")
            return True

        rebuilt = []
        total = len(rebuild_list)
        success = True

        for i, artifact_name in enumerate(rebuild_list, 1):
            logger.info(f"\n[{i}/{total}] Building {artifact_name}...")
            try:
                result = executor.execute_build(artifact_name, step=i, total=total)
                if result.success:
                    rebuilt.append(artifact_name)
                    logger.info(f"  [OK] {artifact_name} built successfully")
                else:
                    success = False
                    logger.error(f"  [FAIL] {artifact_name} build failed: {result.error}")
            except Exception as e:
                success = False
                logger.error(f"  [FAIL] {artifact_name} build failed with exception: {e}")

        return success

    def analyze_status(self) -> Dict[str, Dict]:
        logger.info("Analyzing artifact status...")

        current_fps = self._get_current_dataset_fingerprints()

        results = {}

        ds_map = {
            "question_tag.csv": self.config.dataset.train_csv,
            "eval.csv": self.config.dataset.eval_csv,
        }
        for name, fp in current_fps.items():
            path = ds_map.get(name)
            if fp == "no_matching_columns":
                status_val = "invalid_columns"
            else:
                status_val = "present" if fp not in ('missing', 'unreadable') else fp
            results[name] = {
                "type": "dataset",
                "status": status_val,
                "fingerprint": fp if fp not in ('missing', 'unreadable', 'no_matching_columns') else None,
                "path": str(path) if path is not None else None
            }

        for artifact_name in self.dependency_graph.keys():
            status = self._validate_artifact(artifact_name)
            is_stale = False
            reason = None

            if status == ArtifactStatus.UP_TO_DATE:
                is_stale = self._dependencies_changed(artifact_name, current_fps)
                if is_stale:
                    status = ArtifactStatus.STALE
                    reason = "dependencies changed"
            elif status == ArtifactStatus.PARTIAL:
                reason = "metadata exists but model file missing"

            entry = {
                "type": self.dependency_graph[artifact_name]["type"],
                "status": status.value,
            }
            if reason:
                entry["reason"] = reason
            results[artifact_name] = entry

        return results

    def build_all(
        self,
        force: bool = False,
        dry_run: bool = False,
        artifacts: Optional[List[str]] = None,
        use_lock: bool = True,
        lock_timeout: float = 300.0
    ) -> BuildResult:
        import time
        from .executor import BuildExecutor

        start_time = time.time()

        rebuild_list = self.calculate_rebuild_set(force=force, artifacts=artifacts)

        if not rebuild_list:
            logger.info("All artifacts up-to-date, nothing to build")
            checked_artifacts = self.INFERENCE_ARTIFACTS if self.inference_only else list(self.dependency_graph.keys())
            return BuildResult(
                success=True,
                rebuilt_artifacts=[],
                skipped_artifacts=checked_artifacts,
                duration=time.time() - start_time
            )

        if dry_run:
            logger.info(f"DRY RUN: Would rebuild {len(rebuild_list)} artifacts:")
            for i, artifact in enumerate(rebuild_list, 1):
                logger.info(f"  {i}. {artifact}")
            return BuildResult(
                success=True,
                rebuilt_artifacts=[],
                skipped_artifacts=[a for a in self.dependency_graph.keys() if a not in rebuild_list],
                duration=time.time() - start_time
            )

        lock_file = self.config.output.classifier_dir.parent / ".build.lock"

        if use_lock:
            return self._build_with_lock(
                rebuild_list, force, artifacts, lock_file, lock_timeout, start_time
            )
        else:
            return self._execute_build(rebuild_list, start_time)

    def _build_with_lock(
        self,
        rebuild_list: List[str],
        force: bool,
        artifacts: Optional[List[str]],
        lock_file: Path,
        lock_timeout: float,
        start_time: float
    ) -> BuildResult:
        import time
        with build_lock(lock_file, timeout=lock_timeout):
            rebuild_list = self.calculate_rebuild_set(force=force, artifacts=artifacts)

            if not rebuild_list:
                logger.info("All artifacts now up-to-date (built by another process)")
                checked_artifacts = self.INFERENCE_ARTIFACTS if self.inference_only else list(self.dependency_graph.keys())
                return BuildResult(
                    success=True,
                    rebuilt_artifacts=[],
                    skipped_artifacts=checked_artifacts,
                    duration=time.time() - start_time
                )

            return self._execute_build(rebuild_list, start_time)

    def _execute_build(self, rebuild_list: List[str], start_time: float) -> BuildResult:
        import time
        from .executor import BuildExecutor

        executor = BuildExecutor(
            config=self.config,
            verbose=self.verbose
        )

        rebuilt = []
        failed = {}
        total = len(rebuild_list)

        def _has_failed_dependency(artifact: str) -> Optional[str]:
            deps = self.dependency_graph.get(artifact, {}).get("dependencies", [])
            for dep in deps:
                if dep in self.dependency_graph and dep in failed:
                    return dep
            return None

        for i, artifact_name in enumerate(rebuild_list, 1):
            failed_dep = _has_failed_dependency(artifact_name)
            if failed_dep:
                failed[artifact_name] = f"Skipped: upstream dependency '{failed_dep}' failed"
                logger.warning(f"\n[{i}/{total}] Skipping {artifact_name} (dependency '{failed_dep}' failed)")
                continue

            logger.info(f"\n[{i}/{total}] Building {artifact_name}...")

            try:
                result = executor.execute_build(artifact_name, step=i, total=total)
                if result.success:
                    rebuilt.append(artifact_name)
                    logger.info(f"  [OK] {artifact_name} built successfully")
                else:
                    failed[artifact_name] = result.error or "Build failed"
                    logger.error(f"  [FAIL] {artifact_name} build failed: {result.error}")
            except Exception as e:
                failed[artifact_name] = str(e)
                logger.error(f"  [FAIL] {artifact_name} build failed with exception: {e}")

        skipped = [a for a in self.dependency_graph.keys() if a not in rebuilt and a not in failed]

        return BuildResult(
            success=len(failed) == 0,
            rebuilt_artifacts=rebuilt,
            skipped_artifacts=skipped,
            failed_artifacts=failed,
            duration=time.time() - start_time
        )
