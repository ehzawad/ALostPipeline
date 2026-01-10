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
    
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        fallback_answer: str = "i don't know the answer",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.fallback_answer = fallback_answer
        logger.info(f"ConfidenceRanker initialized (threshold={confidence_threshold})")
    
    def rank(
        self,
        sts_results: Optional[List[Dict[str, Any]]],
        classifier_results: Optional[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
        question: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], ConfidenceTelemetry, List[Dict[str, Any]]]:
        sts_results = sts_results or []
        classifier_results = classifier_results or []
        
        if not classifier_results:
            if sts_results:
                result = self._make_result(sts_results[0], source="sts_fallback")
                telemetry = ConfidenceTelemetry(
                    strategy_used="sts_fallback",
                    classifier_confidence=0.0,
                    sts_similarity=sts_results[0].get("similarity"),
                )
                return [result], telemetry, []
            return [], ConfidenceTelemetry("no_results", 0.0, None), []
        
        top_clf = classifier_results[0]
        clf_conf = top_clf.get("confidence", 0.0)
        top_sts = sts_results[0] if sts_results else None
        sts_sim = top_sts.get("similarity") if top_sts else None
        
        if clf_conf >= self.confidence_threshold:
            result = self._make_result(top_clf, source="classifier_confident")
            telemetry = ConfidenceTelemetry(
                strategy_used="classifier_confident",
                classifier_confidence=clf_conf,
                sts_similarity=sts_sim,
            )
            return self._top_k([result], top_k), telemetry, []
        
        sts_agrees_with_clf2 = False
        if top_sts and len(classifier_results) > 1:
            sts_tag = top_sts.get("tag")
            clf2_tag = classifier_results[1].get("tag")
            if sts_tag == clf2_tag:
                sts_agrees_with_clf2 = True
                result = self._make_result(classifier_results[1], source="sts_confirms_clf2")
                telemetry = ConfidenceTelemetry(
                    strategy_used="sts_confirms_clf2",
                    classifier_confidence=clf_conf,
                    sts_similarity=sts_sim,
                    sts_agrees_with_clf2=True,
                )
                return self._top_k([result], top_k), telemetry, []
        
        result = self._make_result(top_clf, source="classifier_default")
        telemetry = ConfidenceTelemetry(
            strategy_used="classifier_default",
            classifier_confidence=clf_conf,
            sts_similarity=sts_sim,
            sts_agrees_with_clf2=sts_agrees_with_clf2,
        )
        return self._top_k([result], top_k), telemetry, []
    
    def _make_result(self, prediction: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
        score = prediction.get("confidence", prediction.get("similarity", 0.0))
        return {
            "tag": prediction.get("tag"),
            "answer": prediction.get("answer", self.fallback_answer),
            "confidence": score,
            "final_score": score,
            "source": source,
        }
    
    def _top_k(self, results: List[Dict], k: Optional[int]) -> List[Dict]:
        if k is None:
            return results
        return results[:k]
