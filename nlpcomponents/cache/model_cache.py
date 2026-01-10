from loguru import logger
import os
import threading
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from ..config import DEFAULT_EMBEDDING_MODEL
from ..utils.faiss_utils import mps_available

_CACHE_LOCK = threading.RLock()

_SHARED_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_CACHED_MODEL_NAME: Optional[str] = None
_CACHED_DEVICE: Optional[str] = None
_DEFAULT_DEVICE: Optional[str] = None

def _resolve_device_preference() -> Optional[str]:
    env_value = os.environ.get('STS_EMBEDDING_DEVICE')
    if not env_value:
        return None

    preferred = env_value.strip().lower()
    if preferred == 'auto':
        return None

    valid_options = {'cpu', 'cuda', 'mps'}
    if preferred not in valid_options:
        logger.warning(
            f"STS_EMBEDDING_DEVICE={preferred} is invalid. Valid options: cpu, cuda, mps, auto. Falling back to auto."
        )
        return None

    if preferred == 'cuda' and not torch.cuda.is_available():
        logger.warning("STS_EMBEDDING_DEVICE=cuda but CUDA is unavailable. Falling back to auto selection.")
        return None

    if preferred == 'mps' and not mps_available():
        logger.warning("STS_EMBEDDING_DEVICE=mps but MPS backend is unavailable. Falling back to auto selection.")
        return None

    return preferred

def _select_device() -> str:
    preferred = _resolve_device_preference()
    if preferred:
        return preferred

    if torch.cuda.is_available():
        return 'cuda'
    if mps_available():
        return 'mps'
    return 'cpu'

def get_default_device() -> str:
    global _DEFAULT_DEVICE
    with _CACHE_LOCK:
        if _DEFAULT_DEVICE is None:
            _DEFAULT_DEVICE = _select_device()
            logger.info(f"Default device selected: {_DEFAULT_DEVICE}")
        return _DEFAULT_DEVICE

def get_torch_device() -> torch.device:
    return torch.device(get_default_device())

def _load_model_on_device(model_name: str, device: str) -> tuple[SentenceTransformer, str]:
    global _DEFAULT_DEVICE
    try:
        return SentenceTransformer(model_name, device=device), device
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not (device == 'cuda' and 'out of memory' in str(exc).lower()):
            raise

        logger.warning(
            f"CUDA OOM while loading {model_name} on GPU. Falling back to CPU. Set STS_EMBEDDING_DEVICE=cpu to avoid GPU usage."
        )
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        with _CACHE_LOCK:
            _DEFAULT_DEVICE = 'cpu'
        return SentenceTransformer(model_name, device='cpu'), 'cpu'

def get_shared_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    force_reload: bool = False
) -> SentenceTransformer:
    global _SHARED_EMBEDDING_MODEL, _CACHED_MODEL_NAME, _CACHED_DEVICE

    with _CACHE_LOCK:
        needs_reload = (
            force_reload or
            _SHARED_EMBEDDING_MODEL is None or
            _CACHED_MODEL_NAME != model_name
        )

        if needs_reload:
            logger.info(f"Loading shared embedding model: {model_name}")

            if _SHARED_EMBEDDING_MODEL is not None:
                logger.info(f"  Replacing existing model: {_CACHED_MODEL_NAME}")

            requested_device = get_default_device()
            logger.info(f"  Preferred device: {requested_device}")

            model, actual_device = _load_model_on_device(model_name, requested_device)

            _SHARED_EMBEDDING_MODEL = model
            _CACHED_MODEL_NAME = model_name
            _CACHED_DEVICE = actual_device

            logger.info(f"  Model loaded successfully on {actual_device.upper()}")
            logger.info(f"  Max sequence length: {_SHARED_EMBEDDING_MODEL.max_seq_length}")
            logger.info(f"  Embedding dimension: {_SHARED_EMBEDDING_MODEL.get_sentence_embedding_dimension()}")
        else:
            logger.debug(f"Reusing cached embedding model: {model_name} on {_CACHED_DEVICE}")

        return _SHARED_EMBEDDING_MODEL

def clear_model_cache():
    global _SHARED_EMBEDDING_MODEL, _CACHED_MODEL_NAME, _CACHED_DEVICE, _DEFAULT_DEVICE

    with _CACHE_LOCK:
        if _SHARED_EMBEDDING_MODEL is not None:
            logger.info(f"Clearing cached embedding model: {_CACHED_MODEL_NAME}")
            _SHARED_EMBEDDING_MODEL = None
            _CACHED_MODEL_NAME = None
            _CACHED_DEVICE = None
            _DEFAULT_DEVICE = None
        else:
            logger.debug("No cached model to clear")

def is_model_cached() -> bool:
    with _CACHE_LOCK:
        return _SHARED_EMBEDDING_MODEL is not None

def get_cached_model_name() -> Optional[str]:
    with _CACHE_LOCK:
        return _CACHED_MODEL_NAME

def get_cached_device() -> Optional[str]:
    with _CACHE_LOCK:
        return _CACHED_DEVICE

def model_has_native_prompts(model: SentenceTransformer) -> bool:
    return hasattr(model, 'encode_query') and callable(getattr(model, 'encode_query', None))

def encode_queries(
    model: SentenceTransformer,
    texts: list[str],
    use_native: bool = True,
    **kwargs
):
    if use_native and model_has_native_prompts(model):
        return model.encode_query(texts, **kwargs)
    return model.encode(texts, **kwargs)

def encode_documents(
    model: SentenceTransformer,
    texts: list[str],
    use_native: bool = True,
    **kwargs
):
    if use_native and model_has_native_prompts(model):
        return model.encode_document(texts, **kwargs)
    return model.encode(texts, **kwargs)

