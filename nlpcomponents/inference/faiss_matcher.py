from __future__ import annotations

import json
from loguru import logger
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..config import SemanticSearchConfig, EmbeddingPrefixConfig
from ..cache.model_cache import get_shared_embedding_model, encode_queries
from ..utils.faiss_utils import get_faiss, is_gpu_available, index_cpu_to_gpu, get_device_string
from ..utils.path_utils import load_tag_answers
from ..utils.errors import format_missing_artifact_error

if TYPE_CHECKING:
    import faiss

class FaissMatcher:

    def __init__(self, config: SemanticSearchConfig, prefixes_config: Optional[EmbeddingPrefixConfig] = None):
        self.config = config
        self.prefixes = prefixes_config or EmbeddingPrefixConfig()
        
        if config.models_dir is None:
            raise ValueError(
                "SemanticSearchConfig.models_dir cannot be None. "
                "Either use NLPPipelineConfig (which auto-populates paths) or explicitly set models_dir."
            )
        self.models_dir = Path(config.models_dir)

        self.embedding_model = None
        self.indices = {}
        self.question_mapping = None
        self.tag_to_answer = {}
        self.metadata = {}
        self.embedding_dim: Optional[int] = None
        self.use_gpu: bool = True
        self._gpu_indices: Dict[str, bool] = {}

    def initialize(self):
        logger.info("Initializing FAISS matcher...")
        
        self._load_metadata()
        self._validate_prefix_metadata()
        
        logger.info(f"  Loading embedding model: {self.config.embedding_model}")
        logger.info(f"  Native prompts: {self.prefixes.use_native_prompts}, prefixes enabled: {self.prefixes.use_prefixes}")
        if self.prefixes.use_prefixes and self.prefixes.use_instruct_format:
            logger.info(f"  STS query format: Instruct + Query (E5-instruct)")
        
        self.embedding_model = get_shared_embedding_model(self.config.embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"  Embedding dim: {self.embedding_dim}")

        self._load_faiss_indices()
        self._load_question_mapping()
        self._load_answers()
        logger.info("  [OK] FAISS matcher ready")
    
    def _load_metadata(self):
        metadata_file = self.models_dir / "sts_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                self.embedding_dim = self.metadata.get('embedding_dim', 1024)
                logger.info(f"  Embedding dim from metadata: {self.embedding_dim}")
            except Exception as e:
                logger.warning(f"  Could not load metadata: {e}")

    def _validate_prefix_metadata(self):
        if not self.metadata:
            return

        required = ["use_native_prompts", "use_prefixes", "use_instruct_format"]
        missing = [key for key in required if key not in self.metadata]
        if missing:
            logger.warning(
                f"  Prefix metadata missing keys: {missing}. "
                "Rebuild artifacts to enable prefix consistency checks."
            )
            return

        mismatches = []
        artifact_use_native = self.metadata.get("use_native_prompts")
        if artifact_use_native != self.prefixes.use_native_prompts:
            mismatches.append(
                f"use_native_prompts (artifact={artifact_use_native}, config={self.prefixes.use_native_prompts})"
            )
        if artifact_use_native:
            if mismatches:
                raise ValueError(
                    "Embedding prefix config mismatch for FAISS artifacts:\n"
                    + "\n".join(mismatches)
                    + "\nRebuild artifacts with: python -m nlpcomponents.cli train-faiss --force"
                )
            return

        artifact_use_prefixes = self.metadata.get("use_prefixes")
        if artifact_use_prefixes != self.prefixes.use_prefixes:
            mismatches.append(
                f"use_prefixes (artifact={artifact_use_prefixes}, config={self.prefixes.use_prefixes})"
            )
        if not artifact_use_prefixes:
            if mismatches:
                raise ValueError(
                    "Embedding prefix config mismatch for FAISS artifacts:\n"
                    + "\n".join(mismatches)
                    + "\nRebuild artifacts with: python -m nlpcomponents.cli train-faiss --force"
                )
            return

        artifact_use_instruct = self.metadata.get("use_instruct_format")
        if artifact_use_instruct != self.prefixes.use_instruct_format:
            mismatches.append(
                f"use_instruct_format (artifact={artifact_use_instruct}, config={self.prefixes.use_instruct_format})"
            )
        if artifact_use_instruct:
            artifact_task = self.metadata.get("instruct_task")
            if artifact_task is not None and artifact_task != self.prefixes.instruct_task:
                mismatches.append(
                    f"instruct_task (artifact={artifact_task}, config={self.prefixes.instruct_task})"
                )
        else:
            artifact_query_prefix = self.metadata.get("sts_query_prefix", "")
            artifact_passage_prefix = self.metadata.get("sts_passage_prefix", "")
            if artifact_query_prefix != (self.prefixes.sts_query_prefix or ""):
                mismatches.append(
                    f"sts_query_prefix (artifact={artifact_query_prefix}, config={self.prefixes.sts_query_prefix})"
                )
            if artifact_passage_prefix != (self.prefixes.sts_passage_prefix or ""):
                mismatches.append(
                    f"sts_passage_prefix (artifact={artifact_passage_prefix}, config={self.prefixes.sts_passage_prefix})"
                )

        if mismatches:
            raise ValueError(
                "Embedding prefix config mismatch for FAISS artifacts:\n"
                + "\n".join(mismatches)
                + "\nRebuild artifacts with: python -m nlpcomponents.cli train-faiss --force"
            )

    def _load_faiss_indices(self):
        similarity_dir = self.models_dir
        if not similarity_dir.exists():
            raise FileNotFoundError(
                format_missing_artifact_error(
                    "FAISS models directory",
                    similarity_dir,
                    "python -m nlpcomponents.cli train-faiss"
                )
            )

        global_index = similarity_dir / "faiss_index_global.index"
        if not global_index.exists():
            raise FileNotFoundError(
                format_missing_artifact_error(
                    "Global FAISS index",
                    global_index,
                    "python -m nlpcomponents.cli train-faiss"
                )
            )

        faiss_mod = get_faiss()
        cpu_index = faiss_mod.read_index(str(global_index))
        
        if self.use_gpu and is_gpu_available():
            try:
                self.indices['global'], self._gpu_indices['global'] = index_cpu_to_gpu(cpu_index)
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU: {e}, using CPU")
                self.indices['global'] = cpu_index
                self._gpu_indices['global'] = False
                self.use_gpu = False
        else:
            self.indices['global'] = cpu_index
            self._gpu_indices['global'] = False
        
        device_info = "GPU" if self._gpu_indices.get('global') else "CPU"
        logger.info(f"  Loaded global index with {self.indices['global'].ntotal} vectors ({device_info})")
        logger.info(f"  FAISS device: {get_device_string()}")
        

        index_dim = self.indices['global'].d
        if self.embedding_dim is not None and index_dim != self.embedding_dim:
            raise ValueError(
                f"FAISS index dimension mismatch: index has {index_dim} dimensions but "
                f"embedding model produces {self.embedding_dim} dimensions. "
                "This usually means the FAISS index was built with a different embedding model. "
                "Rebuild the index with: python -m nlpcomponents.cli train-faiss --force"
            )
        elif self.embedding_dim is None:
            self.embedding_dim = index_dim
            logger.info(f"  Set embedding_dim from FAISS index: {index_dim}")

    def _load_question_mapping(self):
        mapping_file = self.models_dir / "question_mapping.csv"
        if not mapping_file.exists():
            raise FileNotFoundError(
                format_missing_artifact_error(
                    "Question mapping file",
                    mapping_file,
                    "python -m nlpcomponents.cli train-faiss"
                )
            )

        self.question_mapping = pd.read_csv(mapping_file)
        logger.info(f"  Loaded question mapping ({len(self.question_mapping)} rows)")

    def _load_answers(self):
        self.tag_to_answer = load_tag_answers(self.models_dir, required=False)

    def _encode_query(self, query: str, precomputed_embedding: Optional[np.ndarray] = None) -> np.ndarray:
        if precomputed_embedding is not None:
            if precomputed_embedding.ndim == 0 or precomputed_embedding.size == 0:
                raise ValueError(
                    f"Precomputed embedding cannot be empty or scalar, got shape {precomputed_embedding.shape}"
                )
            
            expected_dim = self.embedding_dim or 1024
            
            if precomputed_embedding.ndim == 1:
                if precomputed_embedding.shape[0] != expected_dim:
                    raise ValueError(
                        f"Precomputed embedding dimension mismatch: expected {expected_dim}, got {precomputed_embedding.shape[0]}"
                    )
                return precomputed_embedding.astype('float32').reshape(1, -1)
            
            elif precomputed_embedding.ndim == 2:
                if precomputed_embedding.shape[0] != 1:
                    raise ValueError(
                        f"Batched embeddings not supported: got shape {precomputed_embedding.shape}. "
                        f"This method expects a single embedding with shape ({expected_dim},) or (1, {expected_dim}). "
                        f"For batch search, call search_global() in a loop."
                    )
                if precomputed_embedding.shape[1] != expected_dim:
                    raise ValueError(
                        f"Precomputed embedding dimension mismatch: expected {expected_dim}, got {precomputed_embedding.shape[1]}"
                    )
                return precomputed_embedding.astype('float32')
            
            else:
                raise ValueError(
                    f"Precomputed embedding has unsupported dimensionality: {precomputed_embedding.ndim}D "
                    f"(shape {precomputed_embedding.shape}). Expected 1D or 2D array."
                )

        query_prefixed = self.prefixes.format_sts_query(query)
        embedding = encode_queries(
            self.embedding_model,
            [query_prefixed],
            use_native=self.prefixes.use_native_prompts,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding[0].astype('float32').reshape(1, -1)

    def _search_index(
        self,
        query: str,
        faiss_index,
        result_indices: Optional[List[int]],
        top_k: int,
        source_index_name: str,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        if faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty, returning no results")
            return []
        
        query_embedding = self._encode_query(query, precomputed_embedding=precomputed_embedding)
        similarities, indices = faiss_index.search(query_embedding, top_k)
        similarities = similarities[0]
        indices = indices[0]

        results = []
        for similarity, idx in zip(similarities, indices):
            if idx < 0:
                continue
            original_idx = result_indices[idx] if result_indices else idx
            if original_idx < 0 or original_idx >= len(self.question_mapping):
                continue

            row = self.question_mapping.iloc[original_idx]
            results.append({
                'question': row['question'],
                'tag': row['tag'],
                'similarity': float(similarity),
                'score': float(similarity),
                'answer': self.tag_to_answer.get(row['tag'], ""),
                'source_index': source_index_name
            })
        return results

    def search_global(
        self,
        query: str,
        top_k: int,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start = time.time()
        results = self._search_index(
            query,
            self.indices['global'],
            None,
            top_k,
            source_index_name='global',
            precomputed_embedding=precomputed_embedding
        )
        total_time = (time.time() - start) * 1000
        metadata = {
            'strategy_used': 'global',
            'indices_queried': ['global'],
            'num_vectors_searched': self.indices['global'].ntotal,
            'search_time_ms': round(total_time, 2),
            'num_results': len(results),
            'used_precomputed_embedding': precomputed_embedding is not None
        }
        return results, metadata

    def compute_density_score(
        self,
        query: str,
        top_k: int = 100,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, Any]]:
        if self.indices['global'].ntotal == 0:
            logger.warning("FAISS index is empty, cannot compute density score")
            return 0.0, {"error": "empty_index", "top_k": 0}
        
        query_embedding = self._encode_query(query, precomputed_embedding)
        
        actual_k = min(top_k, self.indices['global'].ntotal)
        
        similarities, indices = self.indices['global'].search(query_embedding, actual_k)
        similarities = similarities[0]
        
        valid_mask = indices[0] >= 0
        if not np.any(valid_mask):
            return 0.0, {"error": "no_valid_results", "top_k": actual_k}
        similarities = similarities[valid_mask]
        
        mean_sim = float(np.mean(similarities))
        median_sim = float(np.median(similarities))
        min_sim = float(np.min(similarities))
        max_sim = float(np.max(similarities))
        std_sim = float(np.std(similarities))
        
        details = {
            'top_k': actual_k,
            'valid_results': int(np.sum(valid_mask)),
            'mean_similarity': mean_sim,
            'median_similarity': median_sim,
            'min_similarity': min_sim,
            'max_similarity': max_sim,
            'std_similarity': std_sim,
            'top_5_similarities': [float(s) for s in similarities[:5]],
            'percentile_25': float(np.percentile(similarities, 25)),
            'percentile_75': float(np.percentile(similarities, 75)),
        }
        
        return mean_sim, details
