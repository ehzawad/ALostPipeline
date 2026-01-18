from __future__ import annotations

from loguru import logger
import time
from typing import Any, Dict, Optional

from ..config import NLPPipelineConfig
from ..semantic import SemanticSearchEngine
from ..classifier import TagClassifier
from ..ranker import ConfidenceRanker

class NLPPipeline:

    def __init__(self, config: Optional[NLPPipelineConfig | Dict[str, Any]] = None):
        if isinstance(config, NLPPipelineConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = NLPPipelineConfig.from_dict(config)
        else:
            self.config = NLPPipelineConfig()

        self.semantic = SemanticSearchEngine(self.config.semantic, prefixes_config=self.config.prefixes)
        self.classifier = TagClassifier(self.config.classifier, prefixes_config=self.config.prefixes)
        self.ranker = ConfidenceRanker(
            confidence_threshold=self.config.ranker.confidence_threshold,
            abstain_answer=self.config.ranker.abstain_answer,
            fallback_answer=self.config.ranker.fallback_answer or "",
            enable_sts_reranking=self.config.ranker.enable_sts_reranking,
        )
        self._initialized = False

    def initialize(self):
        if self._initialized:
            logger.info("pipelineNLP already initialized")
            return

        logger.info("Initializing pipelineNLP (STS + classifier)...")
        self.semantic.initialize()
        self.classifier.initialize()
        logger.info("[OK] ConfidenceRanker ready")
        self._initialized = True
        logger.info("[OK] pipelineNLP ready")

    def run(
        self,
        question: str,
        fusion_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        fusion_top_k = fusion_top_k or self.config.fusion_top_k
        start_time = time.time()

        sts_state = self.semantic.search(
            question,
            top_k=self.config.semantic.top_k
        )

        if sts_state.is_ood:
            total_time = (time.time() - start_time) * 1000
            
            irrelevant_result = {
                "tag": None,
                "answer": self.config.ranker.abstain_answer,
                "confidence": 0.0,
                "final_score": 0.0,
                "source": "density_ood",
                "density_ood": True,
            }
            
            return {
                "question": question,
                "answer": self.config.ranker.abstain_answer,
                "primary_tag": None,
                "candidates": [irrelevant_result],
                "dropped_candidates": [],
                "signals": {
                    "sts": {
                        "results": sts_state.results,
                        "metadata": sts_state.metadata
                    },
                    "classifier": {
                        "results": [],
                        "metadata": {
                            "skipped": True,
                            "skipped_reason": "density_ood_detected",
                            "triggered_ood": True
                        }
                    }
                },
                "telemetry": {
                    "ranker": {"density_ood": True},
                    "latency_ms": round(total_time, 2),
                    "fusion_top_k": fusion_top_k,
                    "density_score": sts_state.density_score,
                    "density_threshold": self.config.semantic.density_threshold,
                    "density_details": sts_state.density_details,
                }
            }

        clf_state = self.classifier.predict(
            question,
            top_k=self.config.classifier.top_k
        )

        if clf_state.is_ood:
            total_time = (time.time() - start_time) * 1000
            
            irrelevant_result = {
                "tag": None,
                "answer": self.config.ranker.abstain_answer,
                "confidence": 0.0,
                "final_score": 0.0,
                "source": "entropy_ood",
                "entropy_ood": True,
            }
            
            return {
                "question": question,
                "answer": self.config.ranker.abstain_answer,
                "primary_tag": None,
                "candidates": [irrelevant_result],
                "dropped_candidates": [],
                "signals": {
                    "sts": {
                        "results": sts_state.results,
                        "metadata": sts_state.metadata
                    },
                    "classifier": {
                        "results": clf_state.predictions,
                        "metadata": {
                            **clf_state.metadata,
                            "triggered_ood": True
                        }
                    }
                },
                "telemetry": {
                    "ranker": {"entropy_ood": True},
                    "latency_ms": round(total_time, 2),
                    "fusion_top_k": fusion_top_k,
                    "entropy": clf_state.entropy,
                    "normalized_entropy": clf_state.normalized_entropy,
                    "entropy_threshold": self.config.classifier.entropy_threshold,
                    "density_score": sts_state.density_score,
                }
            }

        ranked, telemetry, dropped_candidates = self.ranker.rank(
            sts_state.results,
            clf_state.predictions,
            top_k=fusion_top_k,
            question=question
        )

        best = ranked[0] if ranked else None
        primary_tag = (best or {}).get("tag")
        
        abstained = getattr(telemetry, 'abstained', False)
        if abstained:
            primary_tag = None
            primary_answer = (
                self.config.ranker.abstain_answer
                or self.config.ranker.fallback_answer
                or ""
            )
        else:
            primary_answer = (
                (best or {}).get("answer")
                or (clf_state.predictions[0]["answer"] if clf_state.predictions else None)
                or (sts_state.results[0]["answer"] if sts_state.results else None)
                or self.config.ranker.fallback_answer
                or ""
            )

        total_time = (time.time() - start_time) * 1000

        return {
            "question": question,
            "answer": primary_answer,
            "primary_tag": primary_tag,
            "candidates": ranked,
            "dropped_candidates": dropped_candidates,
            "signals": {
                "sts": {
                    "results": sts_state.results,
                    "metadata": sts_state.metadata
                },
                "classifier": {
                    "results": clf_state.predictions,
                    "metadata": clf_state.metadata
                }
            },
            "telemetry": {
                "ranker": telemetry.__dict__,
                "latency_ms": round(total_time, 2),
                "fusion_top_k": fusion_top_k,
                "density_score": sts_state.density_score,
                "density_threshold": self.config.semantic.density_threshold if self.config.semantic.enable_density_ood else None,
                "entropy": clf_state.entropy,
                "normalized_entropy": clf_state.normalized_entropy,
                "entropy_threshold": self.config.classifier.entropy_threshold if self.config.classifier.enable_entropy_ood else None,
            }
        }
