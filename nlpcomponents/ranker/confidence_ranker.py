from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ConfidenceTelemetry:
    strategy_used: str
    classifier_confidence: float
    sts_similarity: Optional[float]
    sts_agrees_with_clf2: bool = False


class ConfidenceRanker:
    """
    Ranks candidates from classifier and STS signals to produce final predictions.
    
    Uses two distinct answer types:
    - abstain_answer: Used when the system explicitly abstains (OOD, low confidence)
    - fallback_answer: Used when a prediction's answer field is missing/empty
    
    STS Re-ranking Strategy (enable_sts_reranking):
        When the classifier's top prediction has low confidence (below threshold),
        and the STS semantic search agrees with the classifier's SECOND choice,
        we prefer the second choice over the first.
        
        Rationale: When the classifier is uncertain, STS provides a signal about
        which tag is semantically more similar. If STS points to the second-best
        classifier prediction, this cross-signal agreement suggests the second
        choice may actually be correct despite having lower classifier confidence.
        
        This can be disabled via enable_sts_reranking=False if the behavior is
        undesirable for your use case.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        abstain_answer: str = "",
        fallback_answer: str = "",
        enable_sts_reranking: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.abstain_answer = abstain_answer
        self.fallback_answer = fallback_answer
        self.enable_sts_reranking = enable_sts_reranking
        logger.info(f"ConfidenceRanker initialized (threshold={confidence_threshold}, sts_reranking={enable_sts_reranking})")
    
    def rank(
        self,
        sts_results: Optional[List[Dict[str, Any]]],
        classifier_results: Optional[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
        question: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], ConfidenceTelemetry, List[Dict[str, Any]]]:
        """
        Rank candidates from classifier and STS signals.
        
        Returns:
            Tuple of (ranked_results, telemetry, dropped_candidates)
            - ranked_results: The selected candidates
            - telemetry: Information about the ranking decision
            - dropped_candidates: Candidates that were considered but not selected
        """
        sts_results = sts_results or []
        classifier_results = classifier_results or []
        dropped_candidates: List[Dict[str, Any]] = []
        
        if not classifier_results:
            if sts_results:
                result = self._make_result(sts_results[0], source="sts_fallback")
                # Remaining STS results are dropped
                for i, sts in enumerate(sts_results[1:], start=1):
                    dropped_candidates.append(self._make_result(sts, source=f"sts_dropped_{i}"))
                telemetry = ConfidenceTelemetry(
                    strategy_used="sts_fallback",
                    classifier_confidence=0.0,
                    sts_similarity=sts_results[0].get("similarity"),
                )
                return [result], telemetry, dropped_candidates
            return [], ConfidenceTelemetry("no_results", 0.0, None), []
        
        top_clf = classifier_results[0]
        clf_conf = top_clf.get("confidence", 0.0)
        top_sts = sts_results[0] if sts_results else None
        sts_sim = top_sts.get("similarity") if top_sts else None
        
        if clf_conf >= self.confidence_threshold:
            result = self._make_result(top_clf, source="classifier_confident")
            # Remaining classifier results are dropped
            for i, clf in enumerate(classifier_results[1:], start=1):
                dropped_candidates.append(self._make_result(clf, source=f"classifier_dropped_{i}"))
            telemetry = ConfidenceTelemetry(
                strategy_used="classifier_confident",
                classifier_confidence=clf_conf,
                sts_similarity=sts_sim,
            )
            return self._top_k([result], top_k), telemetry, dropped_candidates
        
        # STS re-ranking: prefer classifier's 2nd choice if STS agrees with it
        sts_agrees_with_clf2 = False
        if self.enable_sts_reranking and top_sts and len(classifier_results) > 1:
            sts_tag = top_sts.get("tag")
            clf2_tag = classifier_results[1].get("tag")
            if sts_tag == clf2_tag:
                sts_agrees_with_clf2 = True
                clf1_tag = top_clf.get("tag")
                clf2_conf = classifier_results[1].get("confidence", 0.0)
                logger.debug(
                    f"STS re-ranking triggered: STS agrees with clf #2 ({clf2_tag}) over #1 ({clf1_tag}). "
                    f"clf1_conf={clf_conf:.3f}, clf2_conf={clf2_conf:.3f}, sts_sim={sts_sim:.3f}"
                )
                result = self._make_result(classifier_results[1], source="sts_confirms_clf2")
                # Top classifier result was dropped in favor of second choice
                dropped_candidates.append(self._make_result(top_clf, source="classifier_dropped_for_sts_confirm"))
                # Remaining classifier results (beyond #2) are also dropped
                for i, clf in enumerate(classifier_results[2:], start=2):
                    dropped_candidates.append(self._make_result(clf, source=f"classifier_dropped_{i}"))
                telemetry = ConfidenceTelemetry(
                    strategy_used="sts_confirms_clf2",
                    classifier_confidence=clf_conf,
                    sts_similarity=sts_sim,
                    sts_agrees_with_clf2=True,
                )
                return self._top_k([result], top_k), telemetry, dropped_candidates
        
        result = self._make_result(top_clf, source="classifier_default")
        # Remaining classifier results are dropped
        for i, clf in enumerate(classifier_results[1:], start=1):
            dropped_candidates.append(self._make_result(clf, source=f"classifier_dropped_{i}"))
        telemetry = ConfidenceTelemetry(
            strategy_used="classifier_default",
            classifier_confidence=clf_conf,
            sts_similarity=sts_sim,
            sts_agrees_with_clf2=sts_agrees_with_clf2,
        )
        return self._top_k([result], top_k), telemetry, dropped_candidates
    
    def _make_result(self, prediction: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
        """Create a standardized result dict from a prediction."""
        score = prediction.get("confidence", prediction.get("similarity", 0.0))
        # Use prediction's answer if available, otherwise fall back to fallback_answer
        answer = prediction.get("answer") or self.fallback_answer
        return {
            "tag": prediction.get("tag"),
            "answer": answer,
            "confidence": score,
            "final_score": score,
            "source": source,
        }
    
    def _top_k(self, results: List[Dict], k: Optional[int]) -> List[Dict]:
        if k is None:
            return results
        return results[:k]
