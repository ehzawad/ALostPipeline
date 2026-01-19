from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, NamedTuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout as FileLockTimeout
from loguru import logger

from ..utils.json_utils import json_default

if TYPE_CHECKING:
    from ..preprocessing.normalizer import TextNormalizer

# Fingerprint length: 16 hex chars = 64 bits, fits in int64 for FAISS IDs
FINGERPRINT_LENGTH = 16


class RowInfo(NamedTuple):
    """Information about a row in the dataset."""
    question: str  # Original question text
    tag: str  # Associated tag
    original_index: int  # Row index in the DataFrame
    normalized_question: str = ""  # Normalized text used for fingerprinting


@dataclass
class ChangeSet:
    new: Set[str]
    deleted: Set[str]
    unchanged: Set[str]
    
    @property
    def total_current(self) -> int:
        return len(self.new) + len(self.unchanged)
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.total_current
        if total == 0:
            return 0.0
        return len(self.unchanged) / total


@dataclass
class CacheMetadata:
    embedding_model: str
    embedding_dim: int
    normalize_embeddings: bool
    prefix_config_hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    # New fields for v2 cache scheme
    cache_version: int = 2  # Version 2 uses normalized question-only fingerprints
    normalizer_config_hash: str = ""  # Hash of normalizer settings
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        return cls(
            embedding_model=data.get("embedding_model", ""),
            embedding_dim=data.get("embedding_dim", 0),
            normalize_embeddings=data.get("normalize_embeddings", True),
            prefix_config_hash=data.get("prefix_config_hash", ""),
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
            cache_version=data.get("cache_version", 1),  # Default to v1 for old caches
            normalizer_config_hash=data.get("normalizer_config_hash", ""),
        )


@dataclass
class CacheStats:
    cache_type: str
    total_entries: int
    total_tags: int
    total_size_bytes: int
    embedding_dim: int
    embedding_model: str
    entries_per_tag: Dict[str, int]
    
    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_type": self.cache_type,
            "total_entries": self.total_entries,
            "total_tags": self.total_tags,
            "total_size_mb": round(self.total_size_mb, 2),
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "entries_per_tag": self.entries_per_tag,
        }


class EmbeddingCacheManager:
    """
    Manages embedding cache with both thread-safety (RLock) and process-safety (FileLock).
    
    The cache stores embeddings organized by tag in .npy files, with an index.json
    tracking fingerprint-to-position mappings. File locking prevents race conditions
    when multiple processes (e.g., training + FAISS building) access the same cache.
    """
    
    # Default timeout for acquiring file lock (5 minutes)
    FILE_LOCK_TIMEOUT = 300.0
    
    def __init__(self, cache_dir: Path, cache_type: str):
        self.cache_dir = Path(cache_dir)
        self.cache_type = cache_type
        self.tags_dir = self.cache_dir / "tags"
        self.index_file = self.cache_dir / "index.json"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.tag_mapping_file = self.cache_dir / "tag_mapping.json"
        self._lock_file = self.cache_dir / ".cache.lock"
        
        # Thread-level lock for in-process safety
        self._lock = threading.RLock()
        self._index: Optional[Dict[str, Any]] = None
        self._metadata: Optional[CacheMetadata] = None
        self._tag_mapping: Optional[Dict[str, Any]] = None
        self._dirty = False
        self._tag_mapping_dirty = False
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tags_dir.mkdir(parents=True, exist_ok=True)
        
        # Process-level file lock for cross-process safety
        self._file_lock = FileLock(str(self._lock_file), timeout=self.FILE_LOCK_TIMEOUT)
        
        logger.debug(f"EmbeddingCacheManager initialized: type={cache_type}, dir={cache_dir}")
    
    @contextmanager
    def _acquire_file_lock(self, operation: str = "cache operation"):
        """Acquire file lock for cross-process safety during write operations."""
        try:
            self._file_lock.acquire()
            logger.debug(f"Acquired file lock for {operation}")
            yield
        except FileLockTimeout:
            raise TimeoutError(
                f"Could not acquire cache lock after {self.FILE_LOCK_TIMEOUT}s for {operation}. "
                f"Another process may be using the cache. Lock file: {self._lock_file}"
            )
        finally:
            try:
                self._file_lock.release()
                logger.debug(f"Released file lock for {operation}")
            except Exception as e:
                logger.warning(f"Error releasing file lock: {e}")
    
    @staticmethod
    def compute_fingerprint(text: str, tag: str = "", normalizer: Optional["TextNormalizer"] = None) -> str:
        """
        Compute a fingerprint for a question text.
        
        In v2 cache scheme (default), fingerprint is based on normalized question only.
        Tag is ignored for fingerprinting but tracked separately for organization.
        
        Args:
            text: The question text
            tag: Ignored in v2 scheme (kept for backwards compatibility signature)
            normalizer: Optional text normalizer. If provided, text is normalized first.
            
        Returns:
            16-character hex fingerprint (64 bits, fits in int64 for FAISS IDs)
        """
        # Normalize text if normalizer provided
        if normalizer is not None:
            text = normalizer.normalize(text)
        
        # V2 scheme: fingerprint based on question only, not question+tag
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:FINGERPRINT_LENGTH]
    
    @staticmethod
    def fingerprint_to_faiss_id(fingerprint: str) -> int:
        """
        Convert a fingerprint to a FAISS-compatible int64 ID.
        
        Uses first 16 hex chars (64 bits) of fingerprint.
        """
        # Take first 16 hex chars and convert to int
        hex_str = fingerprint[:FINGERPRINT_LENGTH]
        return int(hex_str, 16)
    
    @staticmethod
    def faiss_id_to_fingerprint_prefix(faiss_id: int) -> str:
        """
        Convert a FAISS int64 ID back to fingerprint hex prefix.
        
        Note: This only gives the first 16 chars; the full fingerprint
        must be looked up in the index if needed.
        """
        return format(faiss_id, '016x')
    
    def compute_fingerprints_batch(
        self,
        df: pd.DataFrame,
        question_col: str = "question",
        tag_col: str = "tag",
        normalizer: Optional["TextNormalizer"] = None
    ) -> Dict[str, RowInfo]:
        """
        Compute fingerprints for all rows in a DataFrame.
        
        In v2 cache scheme, fingerprints are based on normalized question text only.
        Duplicate questions (same normalized text) are detected and warned about.
        
        Args:
            df: DataFrame with question and tag columns
            question_col: Name of question column
            tag_col: Name of tag column
            normalizer: Text normalizer for consistent fingerprinting
            
        Returns:
            Dict mapping fingerprint -> RowInfo
        """
        fp_map: Dict[str, RowInfo] = {}
        duplicates: List[Tuple[str, str, str, int, int]] = []  # (fp, tag1, tag2, idx1, idx2)
        
        for idx, row in df.iterrows():
            question = str(row[question_col]) if pd.notna(row[question_col]) else ""
            tag = str(row[tag_col]) if pd.notna(row[tag_col]) else ""
            
            # Normalize question for fingerprinting
            normalized = normalizer.normalize(question) if normalizer else question
            
            # V2 scheme: fingerprint from normalized question only
            fp = self.compute_fingerprint(normalized)
            
            if fp in fp_map:
                existing = fp_map[fp]
                # In v2, same fingerprint means same normalized question
                # This should have been caught by data validator, but log it anyway
                if existing.tag != tag:
                    duplicates.append((fp, existing.tag, tag, existing.original_index, int(idx)))
                # Keep the first occurrence
            else:
                fp_map[fp] = RowInfo(
                    question=question,
                    tag=tag,
                    original_index=int(idx),
                    normalized_question=normalized
                )
        
        if duplicates:
            logger.warning(
                f"Found {len(duplicates)} duplicate questions across different tags. "
                f"This indicates data pollution that should be fixed."
            )
            for fp, tag1, tag2, idx1, idx2 in duplicates[:5]:
                logger.warning(f"  Duplicate: rows {idx1} (tag={tag1}) and {idx2} (tag={tag2})")
            if len(duplicates) > 5:
                logger.warning(f"  ... and {len(duplicates) - 5} more duplicates")
        
        return fp_map
    
    def _load_metadata(self) -> Optional[CacheMetadata]:
        if not self.metadata_file.exists():
            return None
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None
    
    def _save_metadata(self, metadata: CacheMetadata) -> None:
        metadata.last_updated = datetime.now().isoformat()
        temp_file = self.metadata_file.with_suffix('.json.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=json_default)
        temp_file.replace(self.metadata_file)
        self._metadata = metadata
    
    @property
    def metadata(self) -> Optional[CacheMetadata]:
        with self._lock:
            if self._metadata is None:
                self._metadata = self._load_metadata()
            return self._metadata
    
    def validate_metadata(
        self,
        embedding_model: str,
        embedding_dim: int,
        normalize_embeddings: bool,
        prefix_config_hash: str,
        normalizer_config_hash: str = ""
    ) -> bool:
        """
        Validate that current cache metadata matches expected configuration.
        
        Returns True if cache is valid and can be reused, False if it should be cleared.
        """
        current = self.metadata
        if current is None:
            if self.index.get("entries"):
                logger.warning("Cache has entries but missing metadata. Treating as invalid.")
                return False
            logger.debug("No cache metadata found, cache is empty/new")
            return True
        
        # Check cache version - v1 caches need migration
        if current.cache_version < 2:
            logger.info(f"Cache version outdated: v{current.cache_version} -> v2 (requires migration)")
            return False
        
        if current.embedding_model != embedding_model:
            logger.info(f"Embedding model changed: {current.embedding_model} -> {embedding_model}")
            return False
        
        if current.embedding_dim != embedding_dim:
            logger.info(f"Embedding dim changed: {current.embedding_dim} -> {embedding_dim}")
            return False
        
        if current.normalize_embeddings != normalize_embeddings:
            logger.info(f"Normalization changed: {current.normalize_embeddings} -> {normalize_embeddings}")
            return False
        
        if current.prefix_config_hash != prefix_config_hash:
            logger.info(f"Prefix config changed: {current.prefix_config_hash[:16]}... -> {prefix_config_hash[:16]}...")
            return False
        
        # Check normalizer config if provided
        if normalizer_config_hash and current.normalizer_config_hash:
            if current.normalizer_config_hash != normalizer_config_hash:
                logger.info(f"Normalizer config changed: {current.normalizer_config_hash[:16]}... -> {normalizer_config_hash[:16]}...")
                return False
        
        return True
    
    def save_metadata(
        self,
        embedding_model: str,
        embedding_dim: int,
        normalize_embeddings: bool,
        prefix_config_hash: str,
        normalizer_config_hash: str = ""
    ) -> None:
        """Save cache metadata including v2 scheme fields."""
        existing = self.metadata
        is_new = existing is None
        metadata = CacheMetadata(
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            normalize_embeddings=normalize_embeddings,
            prefix_config_hash=prefix_config_hash,
            created_at=existing.created_at if existing else datetime.now().isoformat(),
            cache_version=2,  # Always save as v2
            normalizer_config_hash=normalizer_config_hash,
        )
        self._save_metadata(metadata)
        if is_new:
            logger.info(f"  Cache metadata saved: model={embedding_model}, dim={embedding_dim}, version=2")
    
    def _load_index(self) -> Dict[str, Any]:
        if not self.index_file.exists():
            return {"entries": {}, "tag_counts": {}}
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            return {"entries": {}, "tag_counts": {}}
    
    def _save_index(self) -> None:
        if self._index is None:
            return
        temp_file = self.index_file.with_suffix('.json.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, default=json_default)
        temp_file.replace(self.index_file)
        self._dirty = False
    
    @property
    def index(self) -> Dict[str, Any]:
        with self._lock:
            if self._index is None:
                self._index = self._load_index()
            return self._index
    
    def flush(self) -> None:
        with self._lock:
            if self._dirty and self._index is not None:
                self._save_index()
            if self._tag_mapping_dirty and self._tag_mapping is not None:
                self._save_tag_mapping()
    
    # ==================== Tag Mapping Methods ====================
    
    def _load_tag_mapping(self) -> Dict[str, Any]:
        """Load tag mapping from disk."""
        if not self.tag_mapping_file.exists():
            return {
                "fingerprint_to_tag": {},
                "tag_to_fingerprints": {},
                "fingerprint_to_faiss_id": {},
                "faiss_id_to_fingerprint": {},
            }
        try:
            with open(self.tag_mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load tag mapping: {e}")
            return {
                "fingerprint_to_tag": {},
                "tag_to_fingerprints": {},
                "fingerprint_to_faiss_id": {},
                "faiss_id_to_fingerprint": {},
            }
    
    def _save_tag_mapping(self) -> None:
        """Save tag mapping to disk atomically."""
        if self._tag_mapping is None:
            return
        temp_file = self.tag_mapping_file.with_suffix('.json.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(self._tag_mapping, f, default=json_default)
        temp_file.replace(self.tag_mapping_file)
        self._tag_mapping_dirty = False
    
    @property
    def tag_mapping(self) -> Dict[str, Any]:
        """Get tag mapping, loading from disk if needed."""
        with self._lock:
            if self._tag_mapping is None:
                self._tag_mapping = self._load_tag_mapping()
            return self._tag_mapping
    
    def get_tag_for_fingerprint(self, fingerprint: str) -> Optional[str]:
        """Get the tag associated with a fingerprint."""
        return self.tag_mapping.get("fingerprint_to_tag", {}).get(fingerprint)
    
    def get_fingerprints_for_tag(self, tag: str) -> List[str]:
        """Get all fingerprints associated with a tag."""
        return self.tag_mapping.get("tag_to_fingerprints", {}).get(tag, [])
    
    def get_faiss_id_for_fingerprint(self, fingerprint: str) -> Optional[int]:
        """Get the FAISS ID for a fingerprint."""
        id_str = self.tag_mapping.get("fingerprint_to_faiss_id", {}).get(fingerprint)
        return int(id_str) if id_str is not None else None
    
    def get_fingerprint_for_faiss_id(self, faiss_id: int) -> Optional[str]:
        """Get the fingerprint for a FAISS ID."""
        return self.tag_mapping.get("faiss_id_to_fingerprint", {}).get(str(faiss_id))
    
    def update_tag_mapping(
        self,
        fingerprint: str,
        tag: str,
        faiss_id: Optional[int] = None
    ) -> None:
        """
        Update tag mapping for a fingerprint.
        
        Args:
            fingerprint: The question fingerprint
            tag: The associated tag
            faiss_id: Optional FAISS ID (computed from fingerprint if not provided)
        """
        with self._lock:
            mapping = self.tag_mapping
            
            # Update fingerprint -> tag
            mapping.setdefault("fingerprint_to_tag", {})[fingerprint] = tag
            
            # Update tag -> fingerprints
            tag_fps = mapping.setdefault("tag_to_fingerprints", {})
            if tag not in tag_fps:
                tag_fps[tag] = []
            if fingerprint not in tag_fps[tag]:
                tag_fps[tag].append(fingerprint)
            
            # Update FAISS ID mappings
            if faiss_id is None:
                faiss_id = self.fingerprint_to_faiss_id(fingerprint)
            
            mapping.setdefault("fingerprint_to_faiss_id", {})[fingerprint] = str(faiss_id)
            mapping.setdefault("faiss_id_to_fingerprint", {})[str(faiss_id)] = fingerprint
            
            self._tag_mapping_dirty = True
    
    def update_tag_mappings_batch(
        self,
        fp_map: Dict[str, RowInfo]
    ) -> None:
        """
        Update tag mappings for multiple fingerprints at once.
        
        Args:
            fp_map: Dict mapping fingerprint -> RowInfo
        """
        with self._lock:
            mapping = self.tag_mapping
            fp_to_tag = mapping.setdefault("fingerprint_to_tag", {})
            tag_to_fps = mapping.setdefault("tag_to_fingerprints", {})
            fp_to_faiss = mapping.setdefault("fingerprint_to_faiss_id", {})
            faiss_to_fp = mapping.setdefault("faiss_id_to_fingerprint", {})
            
            for fingerprint, row_info in fp_map.items():
                tag = row_info.tag
                faiss_id = self.fingerprint_to_faiss_id(fingerprint)
                
                # Update all mappings
                fp_to_tag[fingerprint] = tag
                
                if tag not in tag_to_fps:
                    tag_to_fps[tag] = []
                if fingerprint not in tag_to_fps[tag]:
                    tag_to_fps[tag].append(fingerprint)
                
                fp_to_faiss[fingerprint] = str(faiss_id)
                faiss_to_fp[str(faiss_id)] = fingerprint
            
            self._tag_mapping_dirty = True
    
    def remove_from_tag_mapping(self, fingerprints: Set[str]) -> int:
        """
        Remove fingerprints from tag mapping.
        
        Args:
            fingerprints: Set of fingerprints to remove
            
        Returns:
            Number of fingerprints removed
        """
        with self._lock:
            mapping = self.tag_mapping
            fp_to_tag = mapping.get("fingerprint_to_tag", {})
            tag_to_fps = mapping.get("tag_to_fingerprints", {})
            fp_to_faiss = mapping.get("fingerprint_to_faiss_id", {})
            faiss_to_fp = mapping.get("faiss_id_to_fingerprint", {})
            
            removed = 0
            for fp in fingerprints:
                if fp in fp_to_tag:
                    tag = fp_to_tag.pop(fp)
                    if tag in tag_to_fps and fp in tag_to_fps[tag]:
                        tag_to_fps[tag].remove(fp)
                        if not tag_to_fps[tag]:
                            del tag_to_fps[tag]
                    removed += 1
                
                if fp in fp_to_faiss:
                    faiss_id_str = fp_to_faiss.pop(fp)
                    if faiss_id_str in faiss_to_fp:
                        del faiss_to_fp[faiss_id_str]
            
            if removed > 0:
                self._tag_mapping_dirty = True
            
            return removed
    
    def save_tag_mapping(self) -> None:
        """Save tag mapping to disk (with file lock)."""
        with self._acquire_file_lock("save_tag_mapping"):
            with self._lock:
                if self._tag_mapping is not None:
                    self._save_tag_mapping()
    
    def detect_changes(self, current_fingerprints: Set[str]) -> ChangeSet:
        cached_fingerprints = set(self.index.get("entries", {}).keys())
        
        new = current_fingerprints - cached_fingerprints
        deleted = cached_fingerprints - current_fingerprints
        unchanged = current_fingerprints & cached_fingerprints
        
        return ChangeSet(new=new, deleted=deleted, unchanged=unchanged)
    
    def _remove_from_index_no_save(self, fingerprints: Set[str]) -> int:
        """
        Internal method to remove fingerprints from index without saving.
        Caller must hold both thread lock and file lock, and is responsible for saving.
        """
        entries = self.index.get("entries", {})
        tag_counts = self.index.get("tag_counts", {})
        removed = 0
        
        for fp in fingerprints:
            if fp in entries:
                entry = entries.pop(fp)
                tag = entry.get("tag")
                if tag and tag in tag_counts:
                    tag_counts[tag] = max(0, tag_counts[tag] - 1)
                removed += 1
        
        if removed > 0:
            self._dirty = True
        
        return removed
    
    def remove_from_index(self, fingerprints: Set[str]) -> int:
        """Remove fingerprints from the index and tag mapping. Uses file lock for cross-process safety."""
        with self._acquire_file_lock("remove_from_index"):
            with self._lock:
                removed = self._remove_from_index_no_save(fingerprints)
                if removed > 0:
                    self._save_index()
                    # Also remove from tag mapping
                    self.remove_from_tag_mapping(fingerprints)
                    if self._tag_mapping_dirty:
                        self._save_tag_mapping()
                return removed
    
    def _tag_filename(self, tag: str) -> str:
        return hashlib.sha256(tag.encode('utf-8')).hexdigest() + ".npy"
    
    def _tag_filepath(self, tag: str) -> Path:
        return self.tags_dir / self._tag_filename(tag)

    def _atomic_save_npy(self, filepath: Path, embeddings: np.ndarray) -> None:
        temp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")
        temp_filepath.parent.mkdir(parents=True, exist_ok=True)
        # Use a file handle so numpy doesn't append ".npy" to the temp filename.
        with open(temp_filepath, "wb") as f:
            np.save(f, embeddings.astype("float32", copy=False))
        temp_filepath.replace(filepath)

    def load_embeddings_for_tag(self, tag: str) -> Optional[np.ndarray]:
        filepath = self._tag_filepath(tag)
        if not filepath.exists():
            return None
        try:
            return np.load(filepath)
        except Exception as e:
            logger.error(f"Failed to load embeddings for tag '{tag}': {e}")
            return None
    
    def save_embeddings_for_tag(
        self,
        tag: str,
        embeddings: np.ndarray,
        fingerprints: List[str]
    ) -> None:
        """
        Save embeddings for a tag, replacing any existing embeddings.
        
        Uses both thread lock (in-process) and file lock (cross-process) to prevent
        race conditions when multiple processes access the same cache.
        """
        if len(fingerprints) != embeddings.shape[0]:
            raise ValueError(
                f"Fingerprint count ({len(fingerprints)}) doesn't match "
                f"embedding count ({embeddings.shape[0]})"
            )
        
        with self._acquire_file_lock(f"save_embeddings_for_tag({tag})"):
            with self._lock:
                filepath = self._tag_filepath(tag)
                # Write to temp file first, then rename for atomicity
                self._atomic_save_npy(filepath, embeddings)
                
                entries = self.index.setdefault("entries", {})
                tag_counts = self.index.setdefault("tag_counts", {})
                
                for i, fp in enumerate(fingerprints):
                    entries[fp] = {
                        "tag": tag,
                        "position": i,
                    }
                
                tag_counts[tag] = len(fingerprints)
                self._dirty = True
                self._save_index()
    
    def append_embeddings_to_tag(
        self,
        tag: str,
        new_embeddings: np.ndarray,
        new_fingerprints: List[str]
    ) -> None:
        """
        Append new embeddings to an existing tag's cache file.
        
        Uses both thread lock (in-process) and file lock (cross-process) to prevent
        race conditions when multiple processes access the same cache.
        """
        if len(new_fingerprints) != new_embeddings.shape[0]:
            raise ValueError(
                f"Fingerprint count ({len(new_fingerprints)}) doesn't match "
                f"embedding count ({new_embeddings.shape[0]})"
            )
        
        # Acquire file lock for cross-process safety
        with self._acquire_file_lock(f"append_embeddings_to_tag({tag})"):
            with self._lock:
                # Re-load existing embeddings inside lock to ensure consistency
                existing = self.load_embeddings_for_tag(tag)
                
                if existing is not None:
                    start_position = len(existing)
                    combined = np.vstack([existing, new_embeddings])
                else:
                    combined = new_embeddings
                    start_position = 0
                
                # Write to temp file first, then rename for atomicity
                filepath = self._tag_filepath(tag)
                self._atomic_save_npy(filepath, combined)
                
                entries = self.index.setdefault("entries", {})
                tag_counts = self.index.setdefault("tag_counts", {})
                
                for i, fp in enumerate(new_fingerprints):
                    entries[fp] = {
                        "tag": tag,
                        "position": start_position + i,
                    }
                
                tag_counts[tag] = tag_counts.get(tag, 0) + len(new_fingerprints)
                self._dirty = True
                self._save_index()
    
    def save_new_embeddings(
        self,
        new_fingerprints: List[str],
        new_embeddings: np.ndarray,
        fp_map: Dict[str, RowInfo]
    ) -> None:
        """
        Save new embeddings to cache, organized by tag.
        
        Also updates the tag mapping for the new fingerprints.
        """
        fp_list = new_fingerprints
        
        if len(fp_list) != new_embeddings.shape[0]:
            raise ValueError(
                f"Fingerprint count ({len(fp_list)}) doesn't match "
                f"embedding count ({new_embeddings.shape[0]})"
            )
        
        tag_groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for i, fp in enumerate(fp_list):
            tag = fp_map[fp].tag
            tag_groups[tag].append((fp, i))
        
        for tag, fp_indices in tag_groups.items():
            fps = [fp for fp, _ in fp_indices]
            indices = [idx for _, idx in fp_indices]
            tag_embeddings = new_embeddings[indices]
            
            self.append_embeddings_to_tag(tag, tag_embeddings, fps)
        
        # Update tag mapping for new fingerprints
        new_fp_map = {fp: fp_map[fp] for fp in fp_list}
        self.update_tag_mappings_batch(new_fp_map)
        self.save_tag_mapping()
        
        logger.info(f"  Saved {len(fp_list)} embeddings to cache ({len(tag_groups)} tags updated)")
    
    def assemble_embeddings(
        self,
        ordered_fingerprints: List[str],
        embedding_dim: Optional[int] = None
    ) -> np.ndarray:
        n = len(ordered_fingerprints)
        if n == 0:
            dim = embedding_dim or (self.metadata.embedding_dim if self.metadata else 1024)
            return np.zeros((0, dim), dtype='float32')
        
        if embedding_dim is None:
            if self.metadata is None:
                raise ValueError("No metadata found and embedding_dim not provided")
            embedding_dim = self.metadata.embedding_dim
        
        result = np.zeros((n, embedding_dim), dtype='float32')
        entries = self.index.get("entries", {})
        
        tag_groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        missing = []
        
        for i, fp in enumerate(ordered_fingerprints):
            if fp not in entries:
                missing.append(fp)
                continue
            entry = entries[fp]
            tag = entry["tag"]
            position = entry["position"]
            tag_groups[tag].append((i, position))
        
        if missing:
            raise KeyError(
                f"Missing {len(missing)} fingerprints from cache. "
                f"First few: {missing[:5]}"
            )
        
        for tag, positions in tag_groups.items():
            tag_embeddings = self.load_embeddings_for_tag(tag)
            if tag_embeddings is None:
                raise FileNotFoundError(f"Missing embedding file for tag '{tag}'")
            
            for result_idx, cache_pos in positions:
                if cache_pos >= len(tag_embeddings):
                    raise IndexError(
                        f"Position {cache_pos} out of bounds for tag '{tag}' "
                        f"(has {len(tag_embeddings)} embeddings)"
                    )
                result[result_idx] = tag_embeddings[cache_pos]
        
        return result
    
    def clear(self) -> None:
        """Clear all cached embeddings. Uses file lock for cross-process safety."""
        with self._acquire_file_lock("clear"):
            with self._lock:
                if self.tags_dir.exists():
                    shutil.rmtree(self.tags_dir)
                self.tags_dir.mkdir(parents=True, exist_ok=True)
                
                if self.index_file.exists():
                    self.index_file.unlink()
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                if self.tag_mapping_file.exists():
                    self.tag_mapping_file.unlink()
                
                self._index = None
                self._metadata = None
                self._tag_mapping = None
                self._dirty = False
                self._tag_mapping_dirty = False
                
                logger.info(f"Cleared {self.cache_type} embedding cache")
    
    def garbage_collect(
        self,
        current_fingerprints: Optional[Set[str]] = None
    ) -> Tuple[int, int]:
        """
        Garbage collect orphaned embeddings and compact cache files.
        
        Uses file lock for cross-process safety. Only saves the index once at the end
        to avoid redundant I/O operations.
        """
        with self._acquire_file_lock("garbage_collect"):
            with self._lock:
                entries = self.index.get("entries", {})
                
                # Remove orphaned fingerprints (no longer in current dataset)
                if current_fingerprints is not None:
                    deleted = set(entries.keys()) - current_fingerprints
                    # Use internal no-save version to avoid double-save
                    self._remove_from_index_no_save(deleted)
                    entries = self.index.get("entries", {})
                
                tag_entries: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
                for fp, entry in entries.items():
                    tag = entry["tag"]
                    position = entry["position"]
                    tag_entries[tag].append((fp, position))
                
                entries_removed = 0
                bytes_freed = 0
                
                for tag_file in self.tags_dir.glob("*.npy"):
                    tag_hash = tag_file.stem
                    
                    matching_tag = None
                    for tag in tag_entries.keys():
                        if self._tag_filename(tag) == tag_file.name:
                            matching_tag = tag
                            break
                    
                    if matching_tag is None:
                        bytes_freed += tag_file.stat().st_size
                        tag_file.unlink()
                        logger.debug(f"Removed orphaned tag file: {tag_file.name}")
                        continue
                    
                    try:
                        embeddings = np.load(tag_file)
                        original_size = tag_file.stat().st_size
                        original_count = len(embeddings)
                        
                        fps_and_positions = tag_entries[matching_tag]
                        if not fps_and_positions:
                            bytes_freed += original_size
                            tag_file.unlink()
                            continue
                        
                        fps_and_positions.sort(key=lambda x: x[1])
                        
                        valid_positions = [pos for _, pos in fps_and_positions]
                        valid_fps = [fp for fp, _ in fps_and_positions]
                        
                        max_pos = max(valid_positions) if valid_positions else -1
                        if max_pos >= len(embeddings):
                            logger.warning(
                                f"Corrupted cache for tag '{matching_tag}': "
                                f"max_pos={max_pos}, file_len={len(embeddings)}. Removing corrupted entries."
                            )
                            for fp in valid_fps:
                                entries.pop(fp, None)
                            tag_file.unlink()
                            entries_removed += original_count
                            bytes_freed += original_size
                            continue
                        
                        new_embeddings = embeddings[valid_positions]
                        
                        for new_pos, fp in enumerate(valid_fps):
                            entries[fp]["position"] = new_pos
                        
                        # Write to temp file first, then rename for atomicity
                        self._atomic_save_npy(tag_file, new_embeddings)
                        
                        new_size = tag_file.stat().st_size
                        entries_removed += original_count - len(new_embeddings)
                        bytes_freed += original_size - new_size
                        
                    except Exception as e:
                        logger.error(f"Error during GC for {tag_file.name}: {e}")
                
                tag_counts = self.index.setdefault("tag_counts", {})
                for tag, fps in tag_entries.items():
                    tag_counts[tag] = len(fps)
                
                # Only save index once at the end (fixes double-save issue)
                self._dirty = True
                self._save_index()
                
                return entries_removed, bytes_freed
    
    def get_stats(self) -> CacheStats:
        entries = self.index.get("entries", {})
        tag_counts = self.index.get("tag_counts", {})
        
        total_size = 0
        if self.index_file.exists():
            total_size += self.index_file.stat().st_size
        if self.metadata_file.exists():
            total_size += self.metadata_file.stat().st_size
        for tag_file in self.tags_dir.glob("*.npy"):
            total_size += tag_file.stat().st_size
        
        metadata = self.metadata
        
        return CacheStats(
            cache_type=self.cache_type,
            total_entries=len(entries),
            total_tags=len(tag_counts),
            total_size_bytes=total_size,
            embedding_dim=metadata.embedding_dim if metadata else 0,
            embedding_model=metadata.embedding_model if metadata else "",
            entries_per_tag=dict(tag_counts),
        )
    
    def exists(self) -> bool:
        return self.index_file.exists() and len(self.index.get("entries", {})) > 0


def get_prefix_config_hash(prefix_config) -> str:
    """
    Compute hash of ALL prefix config settings.
    Use this for general-purpose caching; prefer specific hash functions
    (get_classifier_prefix_hash, get_sts_prefix_hash) for type-specific caches.
    """
    if hasattr(prefix_config, 'get_cache_key'):
        cache_key = prefix_config.get_cache_key()
    else:
        cache_key = json.dumps({
            'use_native_prompts': getattr(prefix_config, 'use_native_prompts', False),
            'use_prefixes': getattr(prefix_config, 'use_prefixes', True),
            'use_instruct_format': getattr(prefix_config, 'use_instruct_format', True),
            'instruct_task': getattr(prefix_config, 'instruct_task', ''),
        }, sort_keys=True)
    
    return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]


def get_classifier_prefix_hash(prefix_config) -> str:
    """
    Compute hash of classifier-relevant prefix settings only.
    
    The classifier uses format_classifier_queries_batch() which only depends on:
    - use_native_prompts
    - use_prefixes  
    - use_instruct_format
    - instruct_task (only if use_instruct_format is True)
    - classifier_query_prefix (only if use_prefixes is True and use_instruct_format is False)
    
    This avoids unnecessary cache invalidation when STS-only settings change.
    """
    use_native = getattr(prefix_config, 'use_native_prompts', False)
    use_prefixes = getattr(prefix_config, 'use_prefixes', True)
    use_instruct = getattr(prefix_config, 'use_instruct_format', True)
    
    # Build cache key with only relevant fields
    cache_data = {
        'use_native_prompts': use_native,
        'use_prefixes': use_prefixes,
        'use_instruct_format': use_instruct,
    }
    
    # Only include instruct_task if actually used
    if use_prefixes and use_instruct:
        cache_data['instruct_task'] = getattr(prefix_config, 'instruct_task', '')
    
    # Only include classifier_query_prefix if used (prefixes enabled, instruct disabled)
    if use_prefixes and not use_instruct:
        cache_data['classifier_query_prefix'] = getattr(prefix_config, 'classifier_query_prefix', '')
    
    cache_key = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]


def get_sts_prefix_hash(prefix_config) -> str:
    """
    Compute hash of STS-relevant prefix settings only.
    
    STS uses format_sts_passages_batch() for corpus embeddings which only depends on:
    - use_native_prompts
    - use_prefixes
    - use_instruct_format (passages don't use instruct format even when enabled)
    - sts_passage_prefix (only if use_prefixes is True and use_instruct_format is False)
    
    Note: Query formatting at inference time uses format_sts_query() which has different
    settings, but the cache is for passages (corpus), not queries.
    
    This avoids unnecessary cache invalidation when classifier-only settings change.
    """
    use_native = getattr(prefix_config, 'use_native_prompts', False)
    use_prefixes = getattr(prefix_config, 'use_prefixes', True)
    use_instruct = getattr(prefix_config, 'use_instruct_format', True)
    
    # Build cache key with only relevant fields
    cache_data = {
        'use_native_prompts': use_native,
        'use_prefixes': use_prefixes,
        'use_instruct_format': use_instruct,
    }
    
    # Only include sts_passage_prefix if used (prefixes enabled, instruct disabled)
    # Note: When use_instruct_format is True, passages are NOT prefixed (only queries are)
    if use_prefixes and not use_instruct:
        cache_data['sts_passage_prefix'] = getattr(prefix_config, 'sts_passage_prefix', '')
    
    cache_key = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]


def get_normalizer_config_hash(normalizer) -> str:
    """
    Compute hash of text normalizer configuration.
    
    This ensures cache invalidation when normalization settings change,
    since fingerprints are computed on normalized text.
    """
    if normalizer is None:
        return ""
    
    cache_data = {
        'unicode_normalize': getattr(normalizer, 'unicode_normalize', True),
        'case_fold': getattr(normalizer, 'case_fold', True),
        'strip_diacritics': getattr(normalizer, 'strip_diacritics_enabled', False),
        'remove_punctuation': getattr(normalizer, 'remove_punctuation', True),
        'normalize_whitespace': getattr(normalizer, 'normalize_whitespace', True),
    }
    
    cache_key = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]


def detect_cache_version(cache_dir: Path) -> int:
    """
    Detect the cache version from a cache directory.
    
    Returns:
        1 for v1 cache (question+tag fingerprints)
        2 for v2 cache (question-only fingerprints)
        0 if cache doesn't exist or is empty
    """
    metadata_file = cache_dir / "metadata.json"
    if not metadata_file.exists():
        return 0
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata.get('cache_version', 1)  # Default to v1 for old caches
    except Exception:
        return 0


def needs_cache_migration(cache_dir: Path, target_version: int = 2) -> bool:
    """
    Check if a cache directory needs migration to a newer version.
    
    Args:
        cache_dir: Path to cache directory
        target_version: Target cache version (default: 2)
        
    Returns:
        True if cache exists and is older than target version
    """
    current_version = detect_cache_version(cache_dir)
    if current_version == 0:
        return False  # No cache exists, no migration needed
    return current_version < target_version


class CacheMigrationInfo:
    """Information about cache migration status."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.current_version = detect_cache_version(cache_dir)
        self.needs_migration = self.current_version > 0 and self.current_version < 2
        
    def get_status_message(self) -> str:
        if self.current_version == 0:
            return "No existing cache found"
        elif self.needs_migration:
            return f"Cache v{self.current_version} found - migration to v2 required (will trigger full rebuild)"
        else:
            return f"Cache v{self.current_version} - up to date"


def migrate_cache_v1_to_v2(
    old_cache_dir: Path,
    auto_clear: bool = True
) -> bool:
    """
    Migrate a v1 cache to v2 by clearing it (requires full rebuild).
    
    V1 to V2 migration cannot preserve embeddings because:
    - V1 fingerprints: SHA256(question + tag)
    - V2 fingerprints: SHA256(normalized_question)
    
    The fingerprints are fundamentally different, so we must clear and rebuild.
    
    Args:
        old_cache_dir: Path to v1 cache directory
        auto_clear: If True, automatically clear the cache
        
    Returns:
        True if migration was needed and performed
    """
    if not needs_cache_migration(old_cache_dir):
        return False
    
    logger.info(f"Migrating cache from v1 to v2: {old_cache_dir}")
    logger.info("  V1 caches use question+tag fingerprints, V2 uses question-only")
    logger.info("  This requires a full rebuild of embeddings")
    
    if auto_clear:
        # Clear the old cache
        cache_manager = EmbeddingCacheManager(old_cache_dir, "migration")
        cache_manager.clear()
        logger.info("  [OK] V1 cache cleared, ready for v2 rebuild")
        return True
    
    logger.warning("  Cache migration requires clearing - set auto_clear=True to proceed")
    return False
