from .model_cache import (
    get_default_device,
    get_shared_embedding_model,
    get_torch_device,
    clear_model_cache,
    model_has_native_prompts,
    encode_queries,
    encode_documents,
)

from .embedding_cache import (
    EmbeddingCacheManager,
    CacheMetadata,
    CacheStats,
    ChangeSet,
    RowInfo,
    get_prefix_config_hash,
)

__all__ = [
    # Model cache
    'get_default_device',
    'get_shared_embedding_model',
    'get_torch_device',
    'clear_model_cache',
    'model_has_native_prompts',
    'encode_queries',
    'encode_documents',
    # Embedding cache
    'EmbeddingCacheManager',
    'CacheMetadata',
    'CacheStats',
    'ChangeSet',
    'RowInfo',
    'get_prefix_config_hash',
]
