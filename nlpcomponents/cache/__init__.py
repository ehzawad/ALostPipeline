from .model_cache import (
    get_default_device,
    get_shared_embedding_model,
    get_torch_device,
    clear_model_cache,
    model_has_native_prompts,
    encode_queries,
    encode_documents,
)

__all__ = [
    'get_default_device',
    'get_shared_embedding_model',
    'get_torch_device',
    'clear_model_cache',
    'model_has_native_prompts',
    'encode_queries',
    'encode_documents',
]
