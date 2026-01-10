from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger

from ..config import SemanticSearchConfig, EmbeddingPrefixConfig
from ..inference import FaissMatcher

@dataclass
class SemanticSearchState:
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    density_score: Optional[float] = None
    density_details: Optional[Dict[str, Any]] = None
    is_ood: bool = False

class SemanticSearchEngine:

    def __init__(self, config: Optional[SemanticSearchConfig] = None, prefixes_config: Optional[EmbeddingPrefixConfig] = None):
        self.config = config or SemanticSearchConfig()
        self.prefixes = prefixes_config
        self._matcher: Optional[FaissMatcher] = None

    def initialize(self):
        logger.info(f"Initializing semantic search engine (models_dir={self.config.models_dir})")
        self._matcher = FaissMatcher(self.config, prefixes_config=self.prefixes)
        self._matcher.initialize()
        if self.config.enable_density_ood:
            logger.info(f"  Density-based OOD detection enabled (top_k={self.config.density_top_k}, threshold={self.config.density_threshold:.2f})")
        logger.info("[OK] Semantic search ready")

    def search(
        self,
        question: str,
        top_k: Optional[int] = None,
        precomputed_embedding: Optional[Any] = None
    ) -> SemanticSearchState:
        if not self._matcher:
            raise RuntimeError("SemanticSearchEngine not initialized")

        results, metadata = self._matcher.search_global(
            question,
            top_k or self.config.top_k,
            precomputed_embedding=precomputed_embedding
        )
        metadata = {**metadata, "strategy_used": "global"}
        
        density_score = None
        density_details = None
        is_ood = False
        
        if self.config.enable_density_ood:
            density_score, density_details = self._matcher.compute_density_score(
                question,
                top_k=self.config.density_top_k,
                precomputed_embedding=precomputed_embedding
            )
            is_ood = density_score < self.config.density_threshold
            
            if is_ood:
                logger.debug(
                    f"OOD detected: density={density_score:.3f} < threshold={self.config.density_threshold:.3f}"
                )
        
        return SemanticSearchState(
            results=results,
            metadata=metadata,
            density_score=density_score,
            density_details=density_details,
            is_ood=is_ood
        )
