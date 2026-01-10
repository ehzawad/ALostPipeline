from __future__ import annotations

import logging
from loguru import logger
import threading
from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    import faiss

logging.getLogger("faiss.loader").setLevel(logging.WARNING)

_FAISS = None
_FAISS_LOCK = threading.Lock()
_GPU_RESOURCES = None
_GPU_RESOURCES_LOCK = threading.Lock()

def mps_available() -> bool:
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def get_faiss():
    global _FAISS
    if _FAISS is None:
        with _FAISS_LOCK:
            if _FAISS is None:
                import faiss as _faiss_mod
                _FAISS = _faiss_mod
    return _FAISS

def is_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False
    faiss_mod = get_faiss()
    if not hasattr(faiss_mod, 'get_num_gpus'):
        return False
    try:
        return faiss_mod.get_num_gpus() > 0
    except Exception:
        return False

def get_gpu_resources():
    global _GPU_RESOURCES
    if not is_gpu_available():
        return None
    if _GPU_RESOURCES is None:
        with _GPU_RESOURCES_LOCK:
            if _GPU_RESOURCES is None:
                faiss_mod = get_faiss()
                try:
                    _GPU_RESOURCES = faiss_mod.StandardGpuResources()
                    logger.info("FAISS GPU resources initialized")
                except Exception as e:
                    logger.warning(f"Failed to create GPU resources: {e}")
                    return None
    return _GPU_RESOURCES

def index_cpu_to_gpu(index, gpu_id: int = 0) -> Tuple:
    if not is_gpu_available():
        logger.debug("GPU not available, keeping index on CPU")
        return index, False
    resources = get_gpu_resources()
    if resources is None:
        return index, False
    faiss_mod = get_faiss()
    try:
        gpu_index = faiss_mod.index_cpu_to_gpu(resources, gpu_id, index)
        logger.info(f"Moved index to GPU {gpu_id} ({index.ntotal} vectors)")
        return gpu_index, True
    except Exception as e:
        logger.warning(f"Failed to move index to GPU: {e}")
        return index, False

def index_gpu_to_cpu(index):
    faiss_mod = get_faiss()
    if not hasattr(faiss_mod, 'index_gpu_to_cpu'):
        logger.debug("index_gpu_to_cpu not available (faiss-cpu build), returning original")
        return index
    try:
        cpu_index = faiss_mod.index_gpu_to_cpu(index)
        logger.debug("Moved index from GPU to CPU")
        return cpu_index
    except Exception as e:
        logger.error(f"Failed to move index from GPU to CPU: {e}")
        raise RuntimeError(f"Cannot convert GPU index to CPU: {e}") from e

def get_device_string() -> str:
    if is_gpu_available():
        faiss_mod = get_faiss()
        num_gpus = faiss_mod.get_num_gpus()
        try:
            gpu_name = torch.cuda.get_device_name(0)
            return f"CUDA GPU ({num_gpus} available: {gpu_name})"
        except Exception:
            return f"CUDA GPU ({num_gpus} available)"
    if mps_available():
        return "MPS (Apple Metal) - Note: FAISS uses CPU, PyTorch uses MPS"
    return "CPU"
