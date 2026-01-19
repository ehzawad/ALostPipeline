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
    get_classifier_prefix_hash,
    get_sts_prefix_hash,
    get_normalizer_config_hash,
    detect_cache_version,
    needs_cache_migration,
    CacheMigrationInfo,
    migrate_cache_v1_to_v2,
    FINGERPRINT_LENGTH,
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
    'get_classifier_prefix_hash',
    'get_sts_prefix_hash',
    'get_normalizer_config_hash',
    # Cache migration
    'detect_cache_version',
    'needs_cache_migration',
    'CacheMigrationInfo',
    'migrate_cache_v1_to_v2',
    'FINGERPRINT_LENGTH',
]
