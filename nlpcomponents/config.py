from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

from .utils.path_utils import PACKAGE_ROOT

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_E5_INSTRUCT_TASK = "Given a user question, retrieve similar questions, and pick the best one up, STS"


@dataclass
class DatasetConfig:
    train_csv: Path = Path("datasets/sts_train.csv")
    eval_csv: Path = Path("datasets/sts_eval.csv")

    def __post_init__(self):
        self.train_csv = _as_path(self.train_csv, Path("datasets/sts_train.csv"))
        self.eval_csv = _as_path(self.eval_csv, Path("datasets/sts_eval.csv"))


@dataclass
class OutputConfig:
    features_dir: Path = Path("datasets/features")
    classifier_dir: Path = Path("models/tag_classifier")
    semantic_dir: Path = Path("models/semantic")

    def __post_init__(self):
        self.features_dir = _as_path(self.features_dir, Path("datasets/features"))
        self.classifier_dir = _as_path(self.classifier_dir, Path("models/tag_classifier"))
        self.semantic_dir = _as_path(self.semantic_dir, Path("models/semantic"))


@dataclass
class EmbeddingPrefixConfig:
    use_native_prompts: bool = False
    use_prefixes: bool = True
    use_instruct_format: bool = True
    instruct_task: str = DEFAULT_E5_INSTRUCT_TASK
    sts_query_prefix: str = "query: "
    sts_passage_prefix: str = "passage: "
    classifier_query_prefix: str = "query: "

    def _format_instruct_query(self, text: str) -> str:
        return f"Instruct: {self.instruct_task}\nQuery: {text}"

    def _apply_prefix(self, text: str, prefix: str) -> str:
        return f"{prefix}{text}" if prefix else text

    def format_sts_query(self, text: str) -> str:
        if self.use_native_prompts or not self.use_prefixes:
            return text
        if self.use_instruct_format:
            return self._format_instruct_query(text)
        return self._apply_prefix(text, self.sts_query_prefix)

    def format_sts_passage(self, text: str) -> str:
        if self.use_native_prompts or not self.use_prefixes:
            return text
        if self.use_instruct_format:
            return text
        return self._apply_prefix(text, self.sts_passage_prefix)

    def format_sts_passages_batch(self, texts: List[str]) -> List[str]:
        if self.use_native_prompts or not self.use_prefixes:
            return texts
        if self.use_instruct_format:
            return texts
        prefix = self.sts_passage_prefix
        if not prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def format_classifier_query(self, text: str) -> str:
        if self.use_native_prompts or not self.use_prefixes:
            return text
        if self.use_instruct_format:
            return self._format_instruct_query(text)
        return self._apply_prefix(text, self.classifier_query_prefix)

    def format_classifier_queries_batch(self, texts: List[str]) -> List[str]:
        if self.use_native_prompts or not self.use_prefixes:
            return texts
        if self.use_instruct_format:
            return [self._format_instruct_query(t) for t in texts]
        prefix = self.classifier_query_prefix
        if not prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def get_metadata(self) -> Dict[str, Any]:
        if self.use_native_prompts or not self.use_prefixes:
            query_format = "{text}"
            passage_format = "{text}"
        elif self.use_instruct_format:
            query_format = f"Instruct: {self.instruct_task}\\nQuery: {{text}}"
            passage_format = "{text}"
        else:
            query_format = f"{self.sts_query_prefix}{{text}}" if self.sts_query_prefix else "{text}"
            passage_format = f"{self.sts_passage_prefix}{{text}}" if self.sts_passage_prefix else "{text}"
        return {
            'use_native_prompts': self.use_native_prompts,
            'use_prefixes': self.use_prefixes,
            'use_instruct_format': self.use_instruct_format,
            'instruct_task': self.instruct_task,
            'sts_query_prefix': self.sts_query_prefix,
            'sts_passage_prefix': self.sts_passage_prefix,
            'classifier_query_prefix': self.classifier_query_prefix,
            'query_format': query_format,
            'passage_format': passage_format,
        }

    def get_cache_key(self) -> str:
        import json
        return json.dumps({
            'use_native_prompts': self.use_native_prompts,
            'use_prefixes': self.use_prefixes,
            'use_instruct_format': self.use_instruct_format,
            'instruct_task': self.instruct_task,
            'sts_query_prefix': self.sts_query_prefix,
            'sts_passage_prefix': self.sts_passage_prefix,
            'classifier_query_prefix': self.classifier_query_prefix,
        }, sort_keys=True)


def _normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else PACKAGE_ROOT / path


def _as_path(value: Optional[str | Path], default: Path) -> Path:
    raw = default if value is None else Path(value)
    return _normalize_path(raw)


@dataclass
class SemanticSearchConfig:
    models_dir: Optional[Path] = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 3
    strategy: str = "global"
    normalize_embeddings: bool = True
    enable_density_ood: bool = False
    density_top_k: int = 100
    density_threshold: float = 0.75


@dataclass
class TagClassifierConfig:
    models_dir: Optional[Path] = None
    features_dir: Optional[Path] = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 10
    normalize_embeddings: bool = True
    enable_entropy_ood: bool = False
    entropy_threshold: float = 0.8
    min_confidence_threshold: float = 0.5
    entropy_weight: float = 0.6
    confidence_weight: float = 0.4
    combined_ood_threshold: float = 0.5


@dataclass
class RankerConfig:
    confidence_threshold: float = 0.9
    abstain_answer: str = ""
    fallback_answer: Optional[str] = None
    vocab_file: Optional[Path] = None
    ngrams_file: Optional[Path] = None


@dataclass
class NLPPipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prefixes: EmbeddingPrefixConfig = field(default_factory=EmbeddingPrefixConfig)
    semantic: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    classifier: TagClassifierConfig = field(default_factory=TagClassifierConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    fusion_top_k: int = 1
    log_dir: Optional[Path] = None

    def __post_init__(self):
        if self.log_dir is not None:
            self.log_dir = _as_path(self.log_dir, Path("nlpcomponents/logs"))
        self.semantic.models_dir = self.output.semantic_dir
        self.classifier.models_dir = self.output.classifier_dir
        self.classifier.features_dir = self.output.features_dir
        self.ranker.vocab_file = self.output.features_dir / "training_vocabulary.json"
        self.ranker.ngrams_file = self.output.features_dir / "manual_ngrams.json"

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "NLPPipelineConfig":
        if not data:
            return cls()
        return cls(
            dataset=DatasetConfig(**data.get("dataset", {})) if data.get("dataset") else DatasetConfig(),
            output=OutputConfig(**data.get("output", {})) if data.get("output") else OutputConfig(),
            prefixes=EmbeddingPrefixConfig(**data.get("prefixes", {})) if data.get("prefixes") else EmbeddingPrefixConfig(),
            semantic=SemanticSearchConfig(**data.get("semantic", {})) if data.get("semantic") else SemanticSearchConfig(),
            classifier=TagClassifierConfig(**data.get("classifier", {})) if data.get("classifier") else TagClassifierConfig(),
            ranker=RankerConfig(**data.get("ranker", {})) if data.get("ranker") else RankerConfig(),
            fusion_top_k=data.get("fusion_top_k", 1),
            log_dir=data.get("log_dir")
        )
