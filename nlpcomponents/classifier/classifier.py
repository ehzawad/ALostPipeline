from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from loguru import logger

from ..config import TagClassifierConfig, EmbeddingPrefixConfig
from ..inference import TagClassifierMatcher

@dataclass
class TagClassifierState:

    predictions: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    entropy: Optional[float] = None
    normalized_entropy: Optional[float] = None
    is_ood: bool = False

class TagClassifier:

    def __init__(self, config: Optional[TagClassifierConfig] = None, prefixes_config: Optional[EmbeddingPrefixConfig] = None):
        self.config = config or TagClassifierConfig()
        self.prefixes = prefixes_config
        self._matcher: Optional[TagClassifierMatcher] = None

    def initialize(self):
        logger.info(f"Initializing tag classifier (models_dir={self.config.models_dir})")
        self._matcher = TagClassifierMatcher(self.config, prefixes_config=self.prefixes)
        self._matcher.initialize()
        if self.config.enable_entropy_ood:
            logger.info(
                f"  Weighted OOD detection enabled (combined_threshold={self.config.combined_ood_threshold:.2f}, "
                f"entropy_weight={self.config.entropy_weight:.2f}, confidence_weight={self.config.confidence_weight:.2f})"
            )
        logger.info("[OK] Tag classifier ready")

    def predict(
        self,
        question: str,
        top_k: Optional[int] = None,
        precomputed_embedding: Optional[Any] = None
    ) -> TagClassifierState:
        if not self._matcher:
            raise RuntimeError("TagClassifier not initialized")

        response = self._matcher.search(
            query=question,
            top_k=top_k or self.config.top_k,
            precomputed_embedding=precomputed_embedding
        )

        predictions = response.get("results", [])
        metadata = response.get("metadata", {})
        
        entropy = metadata.get("entropy")
        normalized_entropy = metadata.get("normalized_entropy")
        top1_confidence = metadata.get("top1_confidence", 0.0)
        
        is_ood = False
        combined_ood_score = None
        if self.config.enable_entropy_ood:
            entropy_score = normalized_entropy if normalized_entropy is not None else 0.0
            confidence_score = 1.0 - top1_confidence
            
            combined_ood_score = (
                self.config.entropy_weight * entropy_score +
                self.config.confidence_weight * confidence_score
            )
            
            is_ood = combined_ood_score > self.config.combined_ood_threshold
            metadata["combined_ood_score"] = combined_ood_score
            
            if is_ood:
                logger.debug(
                    f"Weighted OOD detected: combined_score={combined_ood_score:.3f} (thr={self.config.combined_ood_threshold:.3f}), "
                    f"entropy={entropy_score:.3f} (w={self.config.entropy_weight:.2f}), conf_inv={confidence_score:.3f} (w={self.config.confidence_weight:.2f})"
                )
        
        return TagClassifierState(
            predictions=predictions,
            metadata=metadata,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            is_ood=is_ood
        )
