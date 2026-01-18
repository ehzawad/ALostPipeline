"""
Embedding Cache Manager

Provides fingerprint-based caching for embeddings to enable O(k) incremental updates
instead of O(n) full re-embedding on each training run.

Key concepts:
- Fingerprint: SHA256 hash of (text + tag) that uniquely identifies a question
- Per-tag storage: Embeddings grouped by tag in separate .npy files
- Change detection: Set operations to find new/deleted/unchanged questions
- Assembly: Reconstruct full embedding matrix from cached pieces
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, NamedTuple

import numpy as np
import pandas as pd
from loguru import logger


CACHE_VERSION = 1


class RowInfo(NamedTuple):
    """Information about a row in the dataset."""
    question: str
    tag: str
    original_index: int


@dataclass
class ChangeSet:
    """Result of change detection between current data and cache."""
    new: Set[str]  # fingerprints in current but not cache (need embedding)
    deleted: Set[str]  # fingerprints in cache but not current (orphaned)
    unchanged: Set[str]  # fingerprints in both (skip embedding)
    
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
    """Metadata for cache invalidation checks."""
    embedding_model: str
    embedding_dim: int
    normalize_embeddings: bool
    prefix_config_hash: str
    version: int = CACHE_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        return cls(
            embedding_model=data.get("embedding_model", ""),
            embedding_dim=data.get("embedding_dim", 0),
            normalize_embeddings=data.get("normalize_embeddings", True),
            prefix_config_hash=data.get("prefix_config_hash", ""),
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class CacheStats:
    """Statistics about the embedding cache."""
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
    Manages embedding cache with fingerprint-based change detection.
    
    Features:
    - Fingerprint-based identification of questions
    - Per-tag storage for efficient incremental updates
    - Automatic cache invalidation on config changes
    - Thread-safe operations
    
    Usage:
        cache = EmbeddingCacheManager(cache_dir, "classifier")
        
        # Validate or clear if config changed
        if not cache.validate_metadata(model_name, dim, normalize, prefix_hash):
            cache.clear()
        
        # Compute fingerprints and detect changes
        fp_map = cache.compute_fingerprints_batch(df)
        changes = cache.detect_changes(set(fp_map.keys()))
        
        # Embed only new questions
        if changes.new:
            new_embeddings = model.encode([fp_map[fp].question for fp in changes.new])
            cache.save_new_embeddings(changes.new, new_embeddings, fp_map)
        
        # Assemble final matrix
        ordered_fps = [cache.compute_fingerprint(q, t) for q, t in zip(questions, tags)]
        embeddings = cache.assemble_embeddings(ordered_fps)
    """
    
    def __init__(self, cache_dir: Path, cache_type: str):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Root directory for this cache type
            cache_type: Either "classifier" or "sts"
        """
        self.cache_dir = Path(cache_dir)
        self.cache_type = cache_type
        self.tags_dir = self.cache_dir / "tags"
        self.index_file = self.cache_dir / "index.json"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self._lock = threading.RLock()
        self._index: Optional[Dict[str, Any]] = None
        self._metadata: Optional[CacheMetadata] = None
        self._dirty = False
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tags_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"EmbeddingCacheManager initialized: type={cache_type}, dir={cache_dir}")
    
    # =========================================================================
    # Fingerprinting
    # =========================================================================
    
    @staticmethod
    def compute_fingerprint(text: str, tag: str) -> str:
        """
        Compute a unique fingerprint for a question.
        
        The fingerprint is a 32-char hex string derived from SHA256 of the
        question text and tag. Same content = same fingerprint.
        Edit one character = completely different fingerprint.
        
        Args:
            text: The question text
            tag: The tag/label for the question
            
        Returns:
            32-character hexadecimal fingerprint
        """
        # Use null byte as separator to avoid collisions
        # e.g., "abc" + "def" != "ab" + "cdef"
        content = f"{text}\x00{tag}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]
    
    def compute_fingerprints_batch(
        self,
        df: pd.DataFrame,
        question_col: str = "question",
        tag_col: str = "tag"
    ) -> Dict[str, RowInfo]:
        """
        Compute fingerprints for all rows in a DataFrame.
        
        Args:
            df: DataFrame with question and tag columns
            question_col: Name of the question column
            tag_col: Name of the tag column
            
        Returns:
            Dict mapping fingerprint -> RowInfo(question, tag, original_index)
        """
        fp_map: Dict[str, RowInfo] = {}
        
        for idx, row in df.iterrows():
            question = str(row[question_col]) if pd.notna(row[question_col]) else ""
            tag = str(row[tag_col]) if pd.notna(row[tag_col]) else ""
            
            fp = self.compute_fingerprint(question, tag)
            fp_map[fp] = RowInfo(question=question, tag=tag, original_index=int(idx))
        
        return fp_map
    
    # =========================================================================
    # Metadata Management
    # =========================================================================
    
    def _load_metadata(self) -> Optional[CacheMetadata]:
        """Load metadata from disk."""
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
        """Save metadata to disk."""
        metadata.last_updated = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        self._metadata = metadata
    
    @property
    def metadata(self) -> Optional[CacheMetadata]:
        """Get current metadata, loading from disk if needed."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata
    
    def validate_metadata(
        self,
        embedding_model: str,
        embedding_dim: int,
        normalize_embeddings: bool,
        prefix_config_hash: str
    ) -> bool:
        """
        Validate that cache metadata matches current configuration.
        
        If metadata doesn't match, the cache is invalid and should be cleared.
        
        Args:
            embedding_model: Name of the embedding model
            embedding_dim: Dimension of embeddings
            normalize_embeddings: Whether embeddings are normalized
            prefix_config_hash: Hash of the prefix configuration
            
        Returns:
            True if cache is valid, False if it should be cleared
        """
        current = self.metadata
        if current is None:
            logger.debug("No cache metadata found, cache is empty/new")
            return True  # Empty cache is valid
        
        # Check version
        if current.version != CACHE_VERSION:
            logger.info(f"Cache version mismatch: {current.version} != {CACHE_VERSION}")
            return False
        
        # Check embedding model
        if current.embedding_model != embedding_model:
            logger.info(f"Embedding model changed: {current.embedding_model} -> {embedding_model}")
            return False
        
        # Check embedding dimension
        if current.embedding_dim != embedding_dim:
            logger.info(f"Embedding dim changed: {current.embedding_dim} -> {embedding_dim}")
            return False
        
        # Check normalization
        if current.normalize_embeddings != normalize_embeddings:
            logger.info(f"Normalization changed: {current.normalize_embeddings} -> {normalize_embeddings}")
            return False
        
        # Check prefix config
        if current.prefix_config_hash != prefix_config_hash:
            logger.info(f"Prefix config changed: {current.prefix_config_hash[:16]}... -> {prefix_config_hash[:16]}...")
            return False
        
        return True
    
    def save_metadata(
        self,
        embedding_model: str,
        embedding_dim: int,
        normalize_embeddings: bool,
        prefix_config_hash: str
    ) -> None:
        """Save metadata for cache validation."""
        existing = self.metadata
        is_new = existing is None
        metadata = CacheMetadata(
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            normalize_embeddings=normalize_embeddings,
            prefix_config_hash=prefix_config_hash,
            version=CACHE_VERSION,
            created_at=existing.created_at if existing else datetime.now().isoformat(),
        )
        self._save_metadata(metadata)
        if is_new:
            logger.info(f"  Cache metadata saved: model={embedding_model}, dim={embedding_dim}")
    
    # =========================================================================
    # Index Management
    # =========================================================================
    
    def _load_index(self) -> Dict[str, Any]:
        """Load index from disk."""
        if not self.index_file.exists():
            return {"version": CACHE_VERSION, "entries": {}, "tag_counts": {}}
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            return {"version": CACHE_VERSION, "entries": {}, "tag_counts": {}}
    
    def _save_index(self) -> None:
        """Save index to disk."""
        if self._index is None:
            return
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self._index, f)
        self._dirty = False
    
    @property
    def index(self) -> Dict[str, Any]:
        """Get current index, loading from disk if needed."""
        with self._lock:
            if self._index is None:
                self._index = self._load_index()
            return self._index
    
    def flush(self) -> None:
        """Flush any pending changes to disk."""
        with self._lock:
            if self._dirty and self._index is not None:
                self._save_index()
    
    # =========================================================================
    # Change Detection
    # =========================================================================
    
    def detect_changes(self, current_fingerprints: Set[str]) -> ChangeSet:
        """
        Detect what has changed between current data and cache.
        
        Args:
            current_fingerprints: Set of fingerprints for current dataset
            
        Returns:
            ChangeSet with new, deleted, and unchanged fingerprints
        """
        cached_fingerprints = set(self.index.get("entries", {}).keys())
        
        new = current_fingerprints - cached_fingerprints
        deleted = cached_fingerprints - current_fingerprints
        unchanged = current_fingerprints & cached_fingerprints
        
        return ChangeSet(new=new, deleted=deleted, unchanged=unchanged)
    
    def remove_from_index(self, fingerprints: Set[str]) -> int:
        """
        Remove fingerprints from the index (mark as orphaned).
        
        The actual embeddings remain in storage and are cleaned up by gc.
        
        Args:
            fingerprints: Set of fingerprints to remove
            
        Returns:
            Number of entries removed
        """
        with self._lock:
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
                self._save_index()
            
            return removed
    
    # =========================================================================
    # Per-Tag Storage
    # =========================================================================
    
    def _tag_filename(self, tag: str) -> str:
        """Get filename for a tag's embeddings (hashed to avoid filesystem issues)."""
        return hashlib.md5(tag.encode('utf-8')).hexdigest()[:16] + ".npy"
    
    def _tag_filepath(self, tag: str) -> Path:
        """Get full path to a tag's embedding file."""
        return self.tags_dir / self._tag_filename(tag)
    
    def load_embeddings_for_tag(self, tag: str) -> Optional[np.ndarray]:
        """
        Load embeddings for a specific tag.
        
        Args:
            tag: The tag name
            
        Returns:
            numpy array of embeddings or None if not found
        """
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
        Save embeddings for a tag, replacing any existing data.
        
        Args:
            tag: The tag name
            embeddings: numpy array of shape (n, embedding_dim)
            fingerprints: List of fingerprints in same order as embeddings
        """
        if len(fingerprints) != embeddings.shape[0]:
            raise ValueError(
                f"Fingerprint count ({len(fingerprints)}) doesn't match "
                f"embedding count ({embeddings.shape[0]})"
            )
        
        with self._lock:
            filepath = self._tag_filepath(tag)
            np.save(filepath, embeddings.astype('float32'))
            
            # Update index
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
        Append new embeddings to an existing tag file.
        
        Args:
            tag: The tag name
            new_embeddings: numpy array of new embeddings
            new_fingerprints: List of fingerprints for new embeddings
        """
        if len(new_fingerprints) != new_embeddings.shape[0]:
            raise ValueError(
                f"Fingerprint count ({len(new_fingerprints)}) doesn't match "
                f"embedding count ({new_embeddings.shape[0]})"
            )
        
        with self._lock:
            existing = self.load_embeddings_for_tag(tag)
            
            if existing is not None:
                # Get current count from index
                entries = self.index.get("entries", {})
                current_count = sum(1 for e in entries.values() if e.get("tag") == tag)
                
                # Concatenate
                combined = np.vstack([existing, new_embeddings])
                start_position = current_count
            else:
                combined = new_embeddings
                start_position = 0
            
            # Save combined embeddings
            filepath = self._tag_filepath(tag)
            np.save(filepath, combined.astype('float32'))
            
            # Update index
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
        new_fingerprints: Set[str],
        new_embeddings: np.ndarray,
        fp_map: Dict[str, RowInfo]
    ) -> None:
        """
        Save new embeddings, grouping by tag for efficient storage.
        
        Args:
            new_fingerprints: Set of fingerprints for new questions
            new_embeddings: Embeddings array in same order as new_fingerprints iteration
            fp_map: Mapping from fingerprint to RowInfo
        """
        # Convert set to list to maintain order
        fp_list = list(new_fingerprints)
        
        if len(fp_list) != new_embeddings.shape[0]:
            raise ValueError(
                f"Fingerprint count ({len(fp_list)}) doesn't match "
                f"embedding count ({new_embeddings.shape[0]})"
            )
        
        # Group by tag
        tag_groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for i, fp in enumerate(fp_list):
            tag = fp_map[fp].tag
            tag_groups[tag].append((fp, i))
        
        # Save each tag group
        for tag, fp_indices in tag_groups.items():
            fps = [fp for fp, _ in fp_indices]
            indices = [idx for _, idx in fp_indices]
            tag_embeddings = new_embeddings[indices]
            
            self.append_embeddings_to_tag(tag, tag_embeddings, fps)
        
        logger.info(f"  Saved {len(fp_list)} embeddings to cache ({len(tag_groups)} tags updated)")
    
    # =========================================================================
    # Assembly
    # =========================================================================
    
    def assemble_embeddings(
        self,
        ordered_fingerprints: List[str],
        embedding_dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Assemble embedding matrix from cache in specified order.
        
        Args:
            ordered_fingerprints: List of fingerprints in desired output order
            embedding_dim: Embedding dimension (uses metadata if not provided)
            
        Returns:
            numpy array of shape (len(ordered_fingerprints), embedding_dim)
        """
        n = len(ordered_fingerprints)
        if n == 0:
            dim = embedding_dim or (self.metadata.embedding_dim if self.metadata else 1024)
            return np.zeros((0, dim), dtype='float32')
        
        # Get embedding dimension from metadata
        if embedding_dim is None:
            if self.metadata is None:
                raise ValueError("No metadata found and embedding_dim not provided")
            embedding_dim = self.metadata.embedding_dim
        
        result = np.zeros((n, embedding_dim), dtype='float32')
        entries = self.index.get("entries", {})
        
        # Group by tag for efficient file access
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
        
        # Load each tag file once and extract needed embeddings
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
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            # Remove tag files
            if self.tags_dir.exists():
                shutil.rmtree(self.tags_dir)
            self.tags_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove index and metadata
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            # Reset in-memory state
            self._index = None
            self._metadata = None
            self._dirty = False
            
            logger.info(f"Cleared {self.cache_type} embedding cache")
    
    def garbage_collect(
        self,
        current_fingerprints: Optional[Set[str]] = None
    ) -> Tuple[int, int]:
        """
        Remove orphaned embeddings from storage.
        
        This rewrites tag files to only include embeddings that are
        still referenced in the index (or in current_fingerprints).
        
        Args:
            current_fingerprints: If provided, also removes entries not in this set
            
        Returns:
            Tuple of (entries_removed, bytes_freed)
        """
        with self._lock:
            entries = self.index.get("entries", {})
            
            # If current fingerprints provided, first remove from index
            if current_fingerprints is not None:
                deleted = set(entries.keys()) - current_fingerprints
                self.remove_from_index(deleted)
                entries = self.index.get("entries", {})
            
            # Group remaining entries by tag
            tag_entries: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
            for fp, entry in entries.items():
                tag = entry["tag"]
                position = entry["position"]
                tag_entries[tag].append((fp, position))
            
            entries_removed = 0
            bytes_freed = 0
            
            # Rewrite each tag file with only valid entries
            for tag_file in self.tags_dir.glob("*.npy"):
                tag_hash = tag_file.stem
                
                # Find which tag this file belongs to
                matching_tag = None
                for tag in tag_entries.keys():
                    if self._tag_filename(tag) == tag_file.name:
                        matching_tag = tag
                        break
                
                if matching_tag is None:
                    # Orphaned file, no entries reference it
                    bytes_freed += tag_file.stat().st_size
                    tag_file.unlink()
                    logger.debug(f"Removed orphaned tag file: {tag_file.name}")
                    continue
                
                # Load and compact
                try:
                    embeddings = np.load(tag_file)
                    original_size = tag_file.stat().st_size
                    original_count = len(embeddings)
                    
                    fps_and_positions = tag_entries[matching_tag]
                    if not fps_and_positions:
                        bytes_freed += original_size
                        tag_file.unlink()
                        continue
                    
                    # Sort by position to maintain order
                    fps_and_positions.sort(key=lambda x: x[1])
                    
                    # Extract valid embeddings and build new index
                    valid_positions = [pos for _, pos in fps_and_positions]
                    valid_fps = [fp for fp, _ in fps_and_positions]
                    
                    # Check for gaps (orphaned embeddings)
                    max_pos = max(valid_positions) if valid_positions else -1
                    if max_pos >= len(embeddings):
                        logger.warning(
                            f"Position out of bounds in tag '{matching_tag}': "
                            f"max_pos={max_pos}, embeddings={len(embeddings)}"
                        )
                        continue
                    
                    new_embeddings = embeddings[valid_positions]
                    
                    # Update index with new positions
                    for new_pos, fp in enumerate(valid_fps):
                        entries[fp]["position"] = new_pos
                    
                    # Save compacted file
                    np.save(tag_file, new_embeddings.astype('float32'))
                    
                    new_size = tag_file.stat().st_size
                    entries_removed += original_count - len(new_embeddings)
                    bytes_freed += original_size - new_size
                    
                except Exception as e:
                    logger.error(f"Error during GC for {tag_file.name}: {e}")
            
            # Update tag counts
            tag_counts = self.index.setdefault("tag_counts", {})
            for tag, fps in tag_entries.items():
                tag_counts[tag] = len(fps)
            
            self._dirty = True
            self._save_index()
            
            return entries_removed, bytes_freed
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> CacheStats:
        """Get statistics about the cache."""
        entries = self.index.get("entries", {})
        tag_counts = self.index.get("tag_counts", {})
        
        # Calculate total size
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
        """Check if cache has any data."""
        return self.index_file.exists() and len(self.index.get("entries", {})) > 0


def get_prefix_config_hash(prefix_config) -> str:
    """
    Compute a hash of the prefix configuration for cache invalidation.
    
    Args:
        prefix_config: An EmbeddingPrefixConfig instance
        
    Returns:
        32-character hexadecimal hash
    """
    # Use the existing get_cache_key method if available
    if hasattr(prefix_config, 'get_cache_key'):
        cache_key = prefix_config.get_cache_key()
    else:
        # Fallback: serialize relevant attributes
        import json
        cache_key = json.dumps({
            'use_native_prompts': getattr(prefix_config, 'use_native_prompts', False),
            'use_prefixes': getattr(prefix_config, 'use_prefixes', True),
            'use_instruct_format': getattr(prefix_config, 'use_instruct_format', True),
            'instruct_task': getattr(prefix_config, 'instruct_task', ''),
        }, sort_keys=True)
    
    return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]
