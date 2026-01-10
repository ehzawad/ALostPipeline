from __future__ import annotations

import json
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..cache.model_cache import get_shared_embedding_model, get_torch_device, encode_queries
from ..config import TagClassifierConfig, EmbeddingPrefixConfig
from ..utils.path_utils import FEATURES_DIR, load_tag_answers
from ..utils.ngram_utils import extract_ngram_words
from ..utils.errors import format_missing_artifact_error
from .unified_tag_classifier import UnifiedTagClassifier

def _safe_load_checkpoint(model_file: Path, device: str, expected_dir: Optional[Path] = None) -> Dict[str, Any]:
    model_file = Path(model_file).resolve()

    if expected_dir is not None:
        expected_dir = Path(expected_dir).resolve()
        try:
            model_file.relative_to(expected_dir)
        except ValueError:
            raise ValueError(
                f"Security: Model file {model_file} is outside expected directory {expected_dir}. "
                "This could indicate a path traversal attempt. "
                "If this is intentional, explicitly pass expected_dir=None."
            )

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    try:
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(checkpoint)}")

    expected_types = {
        'num_tags': (int,),
        'tag_encoder_classes': (list,),
        'pattern_mean': (list, np.ndarray),
        'pattern_std': (list, np.ndarray),
        'model_state_dict': (dict,),
        'pattern_dim': (int,),
    }

    missing_keys = [k for k in expected_types if k not in checkpoint]
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_keys}. "
            "Model file may be corrupted or from an incompatible version."
        )

    for key, valid_types in expected_types.items():
        if not isinstance(checkpoint[key], valid_types):
            raise ValueError(
                f"Checkpoint key '{key}' has unexpected type {type(checkpoint[key])}. "
                "Model file may be corrupted or from an incompatible version."
            )

    return checkpoint

class TagClassifierMatcher:

    def __init__(self, config: TagClassifierConfig, prefixes_config: Optional[EmbeddingPrefixConfig] = None):
        self.config = config
        self.prefixes = prefixes_config or EmbeddingPrefixConfig()
        
        if config.models_dir is None:
            raise ValueError(
                "TagClassifierConfig.models_dir cannot be None. "
                "Either use NLPPipelineConfig (which auto-populates paths) or explicitly set models_dir."
            )
        self.models_dir = Path(config.models_dir)
        self.embedding_model_name = config.embedding_model

        self.device = None
        self.embedding_model = None
        self.model = None

        self.tag_encoder = None
        self.pattern_mean = None
        self.pattern_std = None
        self.tag_patterns = None
        self.tags_sorted = None

        self.tag_to_answer: Dict[str, str] = {}
        self.embedding_dim: Optional[int] = None

    def initialize(self):
        logger.info(f"Loading tag classifier matcher from {self.models_dir}")

        self.device = get_torch_device()
        logger.info(f"  Device: {self.device}")

        logger.info(f"  Loading embedding model: {self.embedding_model_name}")
        logger.info(f"  Native prompts: {self.prefixes.use_native_prompts}, prefixes enabled: {self.prefixes.use_prefixes}")
        if self.prefixes.use_prefixes and self.prefixes.use_instruct_format:
            logger.info(f"  Classifier query format: Instruct + Query (E5-instruct)")
        
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)

        model_file = self.models_dir / "unified_tag_classifier.pth"
        if not model_file.exists():
            raise FileNotFoundError(
                format_missing_artifact_error(
                    "Tag classifier model",
                    model_file,
                    "python -m nlpcomponents.cli train-classifier"
                )
            )

        checkpoint = _safe_load_checkpoint(model_file, self.device, expected_dir=self.models_dir)
        self._validate_prefix_metadata(checkpoint)

        required_keys = ['num_tags', 'tag_encoder_classes', 'pattern_mean', 'pattern_std', 
                         'model_state_dict', 'pattern_dim']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(
                f"Checkpoint at {model_file} is missing required keys: {missing_keys}. "
                "This may indicate a corrupted or incompatible model file. "
                "Try rebuilding with: python -m nlpcomponents.cli train-classifier --force"
            )

        num_tags = checkpoint['num_tags']
        self.tag_encoder = checkpoint['tag_encoder_classes']
        self.pattern_mean = np.array(checkpoint['pattern_mean'], dtype=np.float32).squeeze()
        self.pattern_std = np.array(checkpoint['pattern_std'], dtype=np.float32).squeeze()
        
        contrastive_weight = checkpoint.get('contrastive_weight', 0.0)
        if contrastive_weight > 0:
            logger.info(f"  Trained with contrastive loss (weight={contrastive_weight:.2f})")

        if self.pattern_mean.ndim != 1:
            self.pattern_mean = self.pattern_mean.reshape(-1)
        if self.pattern_std.ndim != 1:
            self.pattern_std = self.pattern_std.reshape(-1)

        PATTERN_STD_EPSILON = checkpoint.get('pattern_std_epsilon', 1e-7)
        zero_std = self.pattern_std < PATTERN_STD_EPSILON
        if np.any(zero_std):
            logger.warning(
                f"Pattern std contained {int(np.sum(zero_std))} near-zero values "
                f"(possible checkpoint corruption or old format); replacing with {PATTERN_STD_EPSILON}"
            )
            self.pattern_std[zero_std] = PATTERN_STD_EPSILON

        logger.info(f"  Tags: {num_tags}")
        self._load_ngram_patterns()

        from ..utils.constants import NGRAM_TYPES
        pattern_dim = len(self.tags_sorted) * NGRAM_TYPES
        logger.info(f"  Pattern dim: {pattern_dim} ({NGRAM_TYPES} types × {len(self.tags_sorted)} tags)")

        checkpoint_pattern_dim = checkpoint['pattern_dim']
        if checkpoint_pattern_dim != pattern_dim:
            expected_tags = checkpoint_pattern_dim // NGRAM_TYPES
            actual_tags = len(self.tags_sorted)
            raise ValueError(
                f"Pattern dimension mismatch: checkpoint expects {checkpoint_pattern_dim} "
                f"({expected_tags} tags × {NGRAM_TYPES}), but loaded n-grams have {pattern_dim} "
                f"({actual_tags} tags × {NGRAM_TYPES}). Rebuild artifacts with: python -m nlpcomponents.cli train-classifier --force"
            )

        checkpoint_tags = checkpoint.get('tag_encoder_classes')
        if checkpoint_tags is not None:
            checkpoint_tags_set = set(checkpoint_tags)
            ngram_tags_set = set(self.tags_sorted)

            missing_in_ngrams = checkpoint_tags_set - ngram_tags_set
            extra_in_ngrams = ngram_tags_set - checkpoint_tags_set

            if missing_in_ngrams or extra_in_ngrams:
                error_parts = []
                if missing_in_ngrams:
                    error_parts.append(f"missing from ngrams: {sorted(missing_in_ngrams)[:5]}")
                if extra_in_ngrams:
                    error_parts.append(f"extra in ngrams: {sorted(extra_in_ngrams)[:5]}")
                raise ValueError(
                    f"Tag mismatch between checkpoint and ngrams file. {'; '.join(error_parts)}. "
                    "This can cause silent prediction errors. Rebuild with: python -m nlpcomponents.cli build --force"
                )
        

        checkpoint_embedding_dim = checkpoint.get('embedding_dim')
        model_embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        if checkpoint_embedding_dim is not None and checkpoint_embedding_dim != model_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: checkpoint was trained with {checkpoint_embedding_dim} dims "
                f"but current model '{self.embedding_model_name}' produces {model_embedding_dim} dims. "
                "Either use the same embedding model or rebuild with: python -m nlpcomponents.cli train-classifier --force"
            )
        
        self.embedding_dim = checkpoint_embedding_dim or model_embedding_dim
        if checkpoint_embedding_dim is None:
            logger.warning(
                f"Checkpoint missing 'embedding_dim'; falling back to model dimension {model_embedding_dim}. "
                "Consider rebuilding the model to save embedding_dim explicitly."
            )
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        
        dropout = checkpoint.get('dropout', 0.5)
        
        self.model = UnifiedTagClassifier(
            embedding_dim=self.embedding_dim,
            pattern_dim=pattern_dim,
            num_tags=num_tags,
            dropout=dropout
        ).to(self.device)
        
        load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        CRITICAL_KEYS = {
            'fusion_fc1.weight', 'fusion_fc2.weight', 'output.weight',
            'emb_fc1.weight', 'emb_fc2.weight',
            'pattern_fc1.weight', 'pattern_fc2.weight'
        }
        critical_missing = CRITICAL_KEYS & set(load_result.missing_keys)
        if critical_missing:
            raise RuntimeError(
                f"Critical model weights missing from checkpoint: {sorted(critical_missing)}. "
                "Model would use random weights. Rebuild with: python -m nlpcomponents.cli train-classifier --force"
            )

        if load_result.missing_keys:
            logger.warning(
                f"Checkpoint missing {len(load_result.missing_keys)} non-critical keys "
                f"(using random init): {load_result.missing_keys}. "
                "Consider rebuilding if accuracy is degraded."
            )
        if load_result.unexpected_keys:
            logger.warning(f"Checkpoint has unexpected keys (ignored): {load_result.unexpected_keys[:5]}...")
        self.model.eval()

        self._load_answers()
        logger.info("  Tag classifier matcher ready")

    def _load_answers(self):
        self.tag_to_answer = load_tag_answers(self.models_dir, required=False)

    def _validate_prefix_metadata(self, checkpoint: Dict[str, Any]):
        required = ["use_native_prompts", "use_prefixes", "use_instruct_format"]
        missing = [key for key in required if key not in checkpoint]
        if missing:
            logger.warning(
                f"  Prefix metadata missing keys: {missing}. "
                "Rebuild the classifier to enable prefix consistency checks."
            )
            return

        mismatches = []
        artifact_use_native = checkpoint.get("use_native_prompts")
        if artifact_use_native != self.prefixes.use_native_prompts:
            mismatches.append(
                f"use_native_prompts (artifact={artifact_use_native}, config={self.prefixes.use_native_prompts})"
            )
        if artifact_use_native:
            if mismatches:
                raise ValueError(
                    "Embedding prefix config mismatch for classifier artifacts:\n"
                    + "\n".join(mismatches)
                    + "\nRebuild artifacts with: python -m nlpcomponents.cli train-classifier --force"
                )
            return

        artifact_use_prefixes = checkpoint.get("use_prefixes")
        if artifact_use_prefixes != self.prefixes.use_prefixes:
            mismatches.append(
                f"use_prefixes (artifact={artifact_use_prefixes}, config={self.prefixes.use_prefixes})"
            )
        if not artifact_use_prefixes:
            if mismatches:
                raise ValueError(
                    "Embedding prefix config mismatch for classifier artifacts:\n"
                    + "\n".join(mismatches)
                    + "\nRebuild artifacts with: python -m nlpcomponents.cli train-classifier --force"
                )
            return

        artifact_use_instruct = checkpoint.get("use_instruct_format")
        if artifact_use_instruct != self.prefixes.use_instruct_format:
            mismatches.append(
                f"use_instruct_format (artifact={artifact_use_instruct}, config={self.prefixes.use_instruct_format})"
            )
        if artifact_use_instruct:
            artifact_task = checkpoint.get("instruct_task")
            if artifact_task is not None and artifact_task != self.prefixes.instruct_task:
                mismatches.append(
                    f"instruct_task (artifact={artifact_task}, config={self.prefixes.instruct_task})"
                )
        else:
            artifact_prefix = checkpoint.get("classifier_query_prefix", "")
            if artifact_prefix != (self.prefixes.classifier_query_prefix or ""):
                mismatches.append(
                    f"classifier_query_prefix (artifact={artifact_prefix}, config={self.prefixes.classifier_query_prefix})"
                )

        if mismatches:
            raise ValueError(
                "Embedding prefix config mismatch for classifier artifacts:\n"
                + "\n".join(mismatches)
                + "\nRebuild artifacts with: python -m nlpcomponents.cli train-classifier --force"
            )

    def _load_ngram_patterns(self):
        features_dir = self.config.features_dir or FEATURES_DIR
        primary_path = features_dir / "manual_ngrams.json"
        co_located_path = self.models_dir.parent.parent / "datasets" / "features" / "manual_ngrams.json"
        
        if primary_path.exists():
            path = primary_path
        elif co_located_path.exists():
            logger.warning(
                f"N-gram features not found at configured path {primary_path}, "
                f"falling back to co-located path {co_located_path}. "
                "This may cause dimension mismatches if features were generated for a different layout. "
                "Consider setting features_dir explicitly or regenerating features."
            )
            path = co_located_path
        else:
            raise FileNotFoundError(
                f"N-gram file not found. Checked: {primary_path} and {co_located_path}. "
                "Generate it by running: python -m nlpcomponents.cli features"
            )

        logger.info(f"  Loading n-gram features from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            features = json.load(f)

        self.tag_patterns = {}
        for tag_name, tag_data in features["tags"].items():
            self.tag_patterns[tag_name] = {
                "unigrams": set(item["ngram"] for item in tag_data.get("unigrams", [])),
                "bigrams": set(item["ngram"] for item in tag_data.get("bigrams", [])),
                "trigrams": set(item["ngram"] for item in tag_data.get("trigrams", [])),
                "fourgrams": set(item["ngram"] for item in tag_data.get("fourgrams", [])),
                "fivegrams": set(item["ngram"] for item in tag_data.get("fivegrams", []))
            }

        self.tags_sorted = sorted(self.tag_patterns.keys())
        logger.info(f"    Loaded features for {len(self.tags_sorted)} tags")

    def _compute_pattern_features(self, question: str) -> np.ndarray:
        q_unigrams = extract_ngram_words(question, 1)
        q_bigrams = extract_ngram_words(question, 2)
        q_trigrams = extract_ngram_words(question, 3)
        q_fourgrams = extract_ngram_words(question, 4)
        q_fivegrams = extract_ngram_words(question, 5)

        feat = []
        for tag in self.tags_sorted:
            if tag not in self.tag_patterns:
                logger.warning(f"Tag '{tag}' missing from tag_patterns, using zero features")
                feat.extend([0, 0, 0, 0, 0])
                continue
            patterns = self.tag_patterns[tag]
            unigram_matches = len(q_unigrams & patterns["unigrams"])
            bigram_matches = len(q_bigrams & patterns["bigrams"])
            trigram_matches = len(q_trigrams & patterns["trigrams"])
            fourgram_matches = len(q_fourgrams & patterns["fourgrams"])
            fivegram_matches = len(q_fivegrams & patterns["fivegrams"])
            feat.extend([unigram_matches, bigram_matches, trigram_matches, fourgram_matches, fivegram_matches])
        return np.array(feat, dtype=np.float32)

    def predict(
        self,
        question: str,
        top_k: int = 3,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        if self.model.training:
            raise RuntimeError(
                "Model is in training mode but inference was called. "
                "Ensure model.eval() was called during initialization."
            )

        pattern = self._compute_pattern_features(question)
        pattern = (pattern - self.pattern_mean) / self.pattern_std
        pat_tensor = torch.FloatTensor(pattern).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self._prepare_embedding(question, precomputed_embedding)
            emb_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            logits = self.model(emb_tensor, pat_tensor)
            
            probs = F.softmax(logits, dim=1)[0]
            
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs).item()
            max_entropy = np.log(len(probs))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

        predictions = []
        
        if len(top_indices) == 0:
            return {
                'results': [],
                'metadata': {
                    'num_candidates': 0,
                    'uses_unified_embeddings': False,
                    'entropy': None,
                    'normalized_entropy': 1.0,
                    'max_entropy': None,
                    'top1_confidence': 0.0,
                    'confidence_gap': 0.0,
                }
            }
        
        num_tags = len(self.tag_encoder)
        for idx, prob in zip(top_indices, top_probs):
            if idx < 0 or idx >= num_tags:
                logger.warning(f"Tag index {idx} out of bounds (num_tags={num_tags}), skipping")
                continue
            tag = self.tag_encoder[idx]
            predictions.append({
                'tag': tag,
                'confidence': float(prob),
                'answer': self.tag_to_answer.get(tag, "")
            })

        confidence_gap = float(top_probs[0] - top_probs[1]) if len(top_probs) > 1 else float(top_probs[0])

        return {
            'results': predictions,
            'metadata': {
                'num_candidates': len(predictions),
                'uses_unified_embeddings': False,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'max_entropy': float(max_entropy),
                'top1_confidence': float(top_probs[0]) if len(top_probs) > 0 else 0.0,
                'confidence_gap': confidence_gap,
            }
        }

    def predict_with_uncertainty(
        self,
        question: str,
        top_k: int = 3,
        n_samples: int = 10,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        embedding = self._prepare_embedding(question, precomputed_embedding)
        pattern = self._compute_pattern_features(question)
        pattern = (pattern - self.pattern_mean) / self.pattern_std

        emb_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        pat_tensor = torch.FloatTensor(pattern).unsqueeze(0).to(self.device)

        mean_probs, variance, _ = self.model.predict_with_uncertainty(
            emb_tensor, pat_tensor, n_samples=n_samples
        )
        
        mean_probs = mean_probs[0]
        variance = variance[0]
        
        top_probs, top_indices = torch.topk(mean_probs, min(top_k, len(mean_probs)))
        top_variance = variance[top_indices].cpu().numpy()
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        predictions = []
        num_tags = len(self.tag_encoder)
        valid_variances = []
        for idx, prob, var in zip(top_indices, top_probs, top_variance):
            if idx < 0 or idx >= num_tags:
                logger.warning(f"Tag index {idx} out of bounds (num_tags={num_tags}), skipping")
                continue
            tag = self.tag_encoder[idx]
            predictions.append({
                'tag': tag,
                'confidence': float(prob),
                'uncertainty': float(var),
                'answer': self.tag_to_answer.get(tag, "")
            })
            valid_variances.append(var)

        mean_uncertainty = float(np.mean(valid_variances)) if valid_variances else 0.0

        return {
            'results': predictions,
            'metadata': {
                'num_candidates': len(predictions),
                'uses_unified_embeddings': False,
                'mc_samples': n_samples,
                'mean_uncertainty': mean_uncertainty
            }
        }

    def _encode_embedding(self, question: str) -> np.ndarray:
        question_prefixed = self.prefixes.format_classifier_query(question)
        embedding = encode_queries(
            self.embedding_model,
            [question_prefixed],
            use_native=self.prefixes.use_native_prompts,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False
        )[0]
        return embedding

    def search(
        self,
        query: str,
        top_k: int,
        precomputed_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        start = time.time()
        response = self.predict(query, top_k, precomputed_embedding=precomputed_embedding)
        response['metadata']['inference_time_ms'] = round((time.time() - start) * 1000, 2)
        return response

    def _prepare_embedding(
        self,
        question: str,
        precomputed_embedding: Optional[np.ndarray]
    ) -> np.ndarray:
        if precomputed_embedding is not None:
            embedding = np.asarray(precomputed_embedding, dtype=np.float32)
            if embedding.ndim > 1:
                embedding = embedding.reshape(-1)
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Precomputed embedding dim {embedding.shape[0]} does not match expected dim {self.embedding_dim}"
                )
            return embedding

        return self._encode_embedding(question)
