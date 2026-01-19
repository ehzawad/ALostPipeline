import argparse
import os
import torch
import json
from loguru import logger
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlpcomponents.config import EmbeddingPrefixConfig, DEFAULT_E5_INSTRUCT_TASK, DEFAULT_EMBEDDING_MODEL
from nlpcomponents.build.fingerprint import compute_fingerprint
from nlpcomponents.build.data_validator import DataValidator, DataPollutionError
from nlpcomponents.cache.model_cache import get_default_device
from nlpcomponents.cache.embedding_cache import (
    EmbeddingCacheManager, 
    get_prefix_config_hash, 
    get_sts_prefix_hash,
    get_normalizer_config_hash,
    ChangeSet,
)
from nlpcomponents.build.fingerprint import compute_dataset_fingerprint, compute_classifier_fingerprint
from nlpcomponents.preprocessing.normalizer import TextNormalizer, DEFAULT_NORMALIZER
from nlpcomponents.utils.faiss_utils import get_faiss, is_gpu_available, index_cpu_to_gpu, index_gpu_to_cpu, get_device_string
from nlpcomponents.utils.path_utils import DATASETS_DIR, SEMANTIC_MODELS_DIR, CLASSIFIER_MODELS_DIR
from nlpcomponents.utils.json_utils import json_default

if TYPE_CHECKING:
    import faiss

class STSTrainer:

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        normalize_embeddings: bool = True,
        prefixes: Optional[EmbeddingPrefixConfig] = None,
        cache_dir: Optional[Path] = None,
        use_embedding_cache: bool = True,
        normalizer: Optional[TextNormalizer] = None,
    ):
        self.embedding_model_name = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.prefixes = prefixes or EmbeddingPrefixConfig()
        self.cache_dir = cache_dir
        self.use_embedding_cache = use_embedding_cache
        self.normalizer = normalizer or DEFAULT_NORMALIZER
        
        self.embedding_model = None
        self.embedding_dim: Optional[int] = None
        self.use_unified_embeddings = False
        
        # For incremental updates
        self._fp_map: Optional[dict] = None
        self._changes: Optional[ChangeSet] = None

    def load_data(self, train_file: Path):
        logger.info("Loading training data...")

        df = pd.read_csv(train_file)
        logger.info(f"  Loaded {len(df)} questions from {train_file.name}")
        logger.info(f"  Unique tags: {df['tag'].nunique()}")

        logger.info("\n  Sample tag distribution:")
        tag_counts = df['tag'].value_counts().head(10)
        for tag, count in tag_counts.items():
            logger.info(f"    {tag:40s}: {count:5d} questions")

        return df

    def generate_embeddings(self, questions, tags: Optional[list] = None):
        logger.info("\nGenerating embeddings...")
        logger.info(f"  Model: {self.embedding_model_name}")
        logger.info(f"  Questions: {len(questions)}")
        logger.info(f"  Normalize: {self.normalize_embeddings}")
        logger.info(f"  Native prompts: {self.prefixes.use_native_prompts}, prefixes enabled: {self.prefixes.use_prefixes}")
        
        from nlpcomponents.cache.model_cache import get_shared_embedding_model
        
        device = get_default_device()
        logger.info(f"  Device: {device}")
        if device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Version: {torch.version.cuda}")

        logger.info(f"  Loading model on {device}...")
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"  [OK] Model loaded successfully on {device}")
        
        can_use_cache = (
            self.use_embedding_cache and
            self.cache_dir is not None and
            tags is not None and
            len(tags) == len(questions)
        )

        if can_use_cache:
            return self._generate_embeddings_cached(questions, tags, device)
        else:
            if self.use_embedding_cache and tags is None:
                logger.warning("  Embedding cache disabled: tags not provided")
            elif self.use_embedding_cache and self.cache_dir is None:
                logger.warning("  Embedding cache disabled: cache_dir not set")
            return self._generate_raw_embeddings(questions, device)
    
    def _generate_raw_embeddings(self, questions, device: str):
        from nlpcomponents.cache.model_cache import encode_documents
        
        questions_to_encode = self.prefixes.format_sts_passages_batch(questions)

        batch_size = 128 if device == 'cuda' else 32
        logger.info(f"  Encoding questions (this may take several minutes)...")
        logger.info(f"  Batch size: {batch_size}")

        embeddings = encode_documents(
            self.embedding_model,
            questions_to_encode,
            use_native=self.prefixes.use_native_prompts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            batch_size=batch_size
        )

        self.embedding_dim = embeddings.shape[1]
        logger.info(f"  [OK] Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

        return embeddings

    def _generate_embeddings_cached(self, questions, tags, device: str):
        from nlpcomponents.cache.model_cache import encode_documents
        
        cache_path = self.cache_dir / "embeddings" / "sts"
        cache = EmbeddingCacheManager(cache_path, "sts")
        
        logger.info(f"  Cache location: {cache_path}")
        
        # Use STS-specific hash to avoid invalidation when classifier-only settings change
        prefix_hash = get_sts_prefix_hash(self.prefixes)
        normalizer_hash = get_normalizer_config_hash(self.normalizer)
        embedding_dim = self.embedding_dim
        
        if cache.exists():
            if not cache.validate_metadata(
                embedding_model=self.embedding_model_name,
                embedding_dim=embedding_dim,
                normalize_embeddings=self.normalize_embeddings,
                prefix_config_hash=prefix_hash,
                normalizer_config_hash=normalizer_hash
            ):
                logger.info("  Cache invalidated (config changed), clearing...")
                cache.clear()
        
        df = pd.DataFrame({'question': questions, 'tag': tags})
        # Use normalizer for v2 fingerprinting (question-only, normalized)
        fp_map = cache.compute_fingerprints_batch(df, normalizer=self.normalizer)
        
        changes = cache.detect_changes(set(fp_map.keys()))
        
        # Store for incremental FAISS updates
        self._fp_map = fp_map
        self._changes = changes
        
        logger.info(f"  Embedding cache: {len(changes.unchanged)} cached, {len(changes.new)} new, {len(changes.deleted)} orphaned")
        
        if changes.cache_hit_rate > 0:
            logger.info(f"  Cache hit rate: {changes.cache_hit_rate:.1%}")
        
        if changes.new:
            logger.info(f"  Encoding {len(changes.new)} new questions...")
            
            new_fps = sorted(changes.new)
            new_questions = [fp_map[fp].question for fp in new_fps]
            
            questions_to_encode = self.prefixes.format_sts_passages_batch(new_questions)
            batch_size = 128 if device == 'cuda' else 32
            
            new_embeddings = encode_documents(
                self.embedding_model,
                questions_to_encode,
                use_native=self.prefixes.use_native_prompts,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
                batch_size=batch_size
            )
            
            cache.save_new_embeddings(new_fps, new_embeddings, fp_map)
        else:
            logger.info("  All embeddings loaded from cache!")
        
        cache.save_metadata(
            embedding_model=self.embedding_model_name,
            embedding_dim=embedding_dim,
            normalize_embeddings=self.normalize_embeddings,
            prefix_config_hash=prefix_hash,
            normalizer_config_hash=normalizer_hash
        )
        
        if changes.deleted:
            cache.remove_from_index(changes.deleted)
        
        # Use normalized fingerprints (no tag in v2 scheme)
        ordered_fps = [
            cache.compute_fingerprint(self.normalizer.normalize(q))
            for q in questions
        ]
        embeddings = cache.assemble_embeddings(ordered_fps, embedding_dim=embedding_dim)
        
        logger.info(f"  [OK] Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        return embeddings

    def build_faiss_indices(
        self, 
        embeddings, 
        indices_to_build='global', 
        use_gpu: bool = True,
        fingerprints: Optional[list] = None
    ):
        """
        Build FAISS indices with content-hash IDs for incremental updates.
        
        Uses IndexIDMap2 wrapper to support add_with_ids and remove_ids operations.
        
        Args:
            embeddings: Embedding vectors to add
            indices_to_build: Which indices to build (only 'global' supported)
            use_gpu: Whether to use GPU for building
            fingerprints: List of fingerprints for each embedding (for content-hash IDs)
        """
        logger.info("\nBuilding FAISS indices...")
        if self.embedding_dim is None:
            raise ValueError("embedding_dim not set; generate embeddings before building indices.")
        embeddings_f32 = embeddings.astype('float32')

        if indices_to_build not in ('all', 'global', None):
            logger.warning("Class-specific indices are no longer supported; building global index only.")

        logger.info("  Building global index with IndexIDMap2 (supports incremental updates)...")
        logger.info(f"  FAISS device: {get_device_string()}")

        start_time = time.time()
        faiss_mod = get_faiss()
        
        # Use IndexIDMap2 to support content-hash IDs and removal
        base_index = faiss_mod.IndexFlatIP(self.embedding_dim)
        index_global = faiss_mod.IndexIDMap2(base_index)
        
        # Generate FAISS IDs from fingerprints
        if fingerprints is not None:
            ids = np.array([
                EmbeddingCacheManager.fingerprint_to_faiss_id(fp) 
                for fp in fingerprints
            ], dtype=np.int64)
            logger.info(f"  Using content-hash IDs (from {len(fingerprints)} fingerprints)")
        else:
            # Fallback to sequential IDs if no fingerprints provided
            ids = np.arange(len(embeddings_f32), dtype=np.int64)
            logger.warning("  No fingerprints provided, using sequential IDs (incremental updates disabled)")
        
        gpu_used = False
        if use_gpu and is_gpu_available():
            try:
                # For IndexIDMap2, we need to add on CPU then potentially move
                # GPU doesn't support IndexIDMap2 directly, so we build on CPU
                index_global.add_with_ids(embeddings_f32, ids)
                logger.info("  Index built on CPU (IndexIDMap2 requires CPU)")
            except Exception as e:
                logger.warning(f"  Index building failed: {e}")
                raise
        else:
            index_global.add_with_ids(embeddings_f32, ids)
        
        build_time = time.time() - start_time

        device_str = "GPU" if gpu_used else "CPU"
        logger.info(f"  [OK] Global index built on {device_str}: {index_global.ntotal:,} vectors in {build_time:.2f}s")
        logger.info(f"  Index type: IndexIDMap2(IndexFlatIP) - supports incremental updates")
        
        return {'global': index_global}
    
    def load_existing_index(self, index_path: Path) -> Optional["faiss.Index"]:
        """Load an existing FAISS index for incremental updates."""
        if not index_path.exists():
            return None
        
        faiss_mod = get_faiss()
        try:
            index = faiss_mod.read_index(str(index_path))
            logger.info(f"  Loaded existing index: {index.ntotal:,} vectors")
            return index
        except Exception as e:
            logger.warning(f"  Failed to load existing index: {e}")
            return None
    
    def update_index_incremental(
        self,
        index: "faiss.Index",
        changes: ChangeSet,
        new_embeddings: np.ndarray,
        new_fingerprints: list
    ) -> "faiss.Index":
        """
        Incrementally update a FAISS index with changes.
        
        Args:
            index: Existing FAISS IndexIDMap2 index
            changes: ChangeSet with new, deleted, unchanged fingerprints
            new_embeddings: Embeddings for new questions
            new_fingerprints: Fingerprints for new questions
            
        Returns:
            Updated index
        """
        faiss_mod = get_faiss()
        
        # Remove deleted entries
        if changes.deleted:
            delete_ids = np.array([
                EmbeddingCacheManager.fingerprint_to_faiss_id(fp)
                for fp in changes.deleted
            ], dtype=np.int64)
            
            # Create ID selector for removal
            id_selector = faiss_mod.IDSelectorArray(delete_ids)
            removed = index.remove_ids(id_selector)
            logger.info(f"  Removed {removed} deleted vectors from index")
        
        # Add new entries
        if changes.new and len(new_embeddings) > 0:
            new_ids = np.array([
                EmbeddingCacheManager.fingerprint_to_faiss_id(fp)
                for fp in new_fingerprints
            ], dtype=np.int64)
            
            index.add_with_ids(new_embeddings.astype('float32'), new_ids)
            logger.info(f"  Added {len(new_embeddings)} new vectors to index")
        
        logger.info(f"  Index now has {index.ntotal:,} vectors")
        return index

    def save_artifacts(
        self, 
        indices, 
        embeddings, 
        df, 
        output_dir: Path, 
        fingerprint: str | None = None, 
        dependencies: dict | None = None,
        fp_map: dict | None = None
    ):
        """
        Save training artifacts including FAISS indices, embeddings, and metadata.
        
        Also saves a FAISS ID lookup file for mapping between fingerprints and FAISS IDs.
        """
        logger.info("\nSaving artifacts...")
        faiss_mod = get_faiss()

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("  Saving FAISS indices...")
        for name, index in indices.items():
            index_file = output_dir / f"faiss_index_{name}.index"
            faiss_mod.write_index(index, str(index_file))
            file_size_mb = index_file.stat().st_size / (1024 * 1024)
            logger.info(f"    [OK] {name:25s}: {index_file.name} ({file_size_mb:.2f} MB)")

        embeddings_file = output_dir / "sts_embeddings.npy"
        np.save(embeddings_file, embeddings)
        file_size_mb = embeddings_file.stat().st_size / (1024 * 1024)
        logger.info(f"  [OK] Embeddings: {embeddings_file.name} ({file_size_mb:.2f} MB)")

        mapping_file = output_dir / "question_mapping.csv"
        df[['question', 'tag']].to_csv(mapping_file, index=False)
        logger.info(f"  [OK] Question mapping: {mapping_file.name}")
        
        # Save FAISS ID lookup for incremental updates
        if fp_map is not None:
            id_lookup = {
                "id_to_fingerprint": {},
                "fingerprint_to_id": {},
                "fingerprint_to_tag": {},
            }
            for fp, row_info in fp_map.items():
                faiss_id = EmbeddingCacheManager.fingerprint_to_faiss_id(fp)
                id_lookup["id_to_fingerprint"][str(faiss_id)] = fp
                id_lookup["fingerprint_to_id"][fp] = faiss_id
                id_lookup["fingerprint_to_tag"][fp] = row_info.tag
            
            id_lookup_file = output_dir / "faiss_id_lookup.json"
            with open(id_lookup_file, 'w', encoding='utf-8') as f:
                json.dump(id_lookup, f, indent=2, default=json_default)
            logger.info(f"  [OK] FAISS ID lookup: {id_lookup_file.name} ({len(fp_map)} entries)")

        metadata = {
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': self.normalize_embeddings,
            'use_unified_embeddings': self.use_unified_embeddings,
            'num_questions': len(df),
            'num_tags': int(df['tag'].nunique()),
            'num_classes': 0,
            'classes': [],
            'class_sizes': {},
            'created_at': datetime.now().isoformat(),
            'indices': {
                'global': indices['global'].ntotal,
            },
            'index_type': 'IndexIDMap2(IndexFlatIP)',
            'supports_incremental': True,
            'fingerprint': fingerprint,
            'dependencies': dependencies or {},
            **self.prefixes.get_metadata()
        }

        metadata_file = output_dir / "sts_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=json_default)
        logger.info(f"  [OK] Metadata: {metadata_file.name}")

        logger.info(f"\n  All artifacts saved to: {output_dir}")

        return metadata

def train_sts(
    indices_to_build='all',
    force: bool = False,
    train_file: Path = Path("nlpcomponents/datasets/question_tag.csv"),
    models_dir: Path = Path("nlpcomponents/models/semantic"),
    classifier_dir: Path = Path("nlpcomponents/models/tag_classifier"),
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    normalize_embeddings: bool = True,
    prefixes: Optional[EmbeddingPrefixConfig] = None,
    cache_dir: Optional[Path] = None,
    use_embedding_cache: bool = True,
    validate_data: bool = True,
    normalizer: Optional[TextNormalizer] = None
):
    print("=" * 80)
    print("STS TRAINING - BUILD FAISS INDICES")
    print("=" * 80)
    
    normalizer = normalizer or DEFAULT_NORMALIZER

    # Phase 1: Data validation (fail early on data pollution)
    if validate_data:
        print("\nPhase 1: Data Validation")
        print("-" * 40)
        validator = DataValidator(normalizer=normalizer)
        try:
            result = validator.validate_and_fail_on_duplicates(train_file, raise_on_duplicates=True)
            print(f"  [OK] Data validation passed: {result.unique_questions} unique questions")
        except DataPollutionError as e:
            print("\n" + e.validation_result.get_report())
            raise

    trainer = STSTrainer(
        embedding_model=embedding_model,
        normalize_embeddings=normalize_embeddings,
        prefixes=prefixes,
        cache_dir=cache_dir,
        use_embedding_cache=use_embedding_cache,
        normalizer=normalizer,
    )

    metadata_file = models_dir / "sts_metadata.json"
    cache_inputs = [train_file]
    
    normalizer_hash = get_normalizer_config_hash(normalizer)
    cache_extra = json.dumps({
        'embedding_model': trainer.embedding_model_name,
        'normalize_embeddings': trainer.normalize_embeddings,
        'indices_to_build': indices_to_build,
        'embedding_prefix_config': trainer.prefixes.get_cache_key(),
        'normalizer_config_hash': normalizer_hash,
    }, sort_keys=True)
    fingerprint = compute_fingerprint(cache_inputs, cache_extra)

    if not force and metadata_file.exists():
        try:
            with metadata_file.open('r', encoding='utf-8') as handle:
                existing = json.load(handle)
        except json.JSONDecodeError:
            existing = {}

        required_files = [
            models_dir / "faiss_index_global.index",
            models_dir / "sts_embeddings.npy",
            models_dir / "question_mapping.csv",
        ]
        if existing.get('fingerprint') == fingerprint and all(path.exists() for path in required_files):
            print("No changes detected; reusing cached FAISS artifacts. Use --force to rebuild.")
            return existing

    df = trainer.load_data(train_file)

    embeddings = trainer.generate_embeddings(
        df['question'].tolist(),
        df['tag'].tolist()
    )
    
    # Get fingerprints for FAISS index IDs
    fp_map = trainer._fp_map
    if fp_map is not None:
        # Get fingerprints in same order as embeddings
        fingerprints = [
            EmbeddingCacheManager.compute_fingerprint(normalizer.normalize(q))
            for q in df['question'].tolist()
        ]
    else:
        fingerprints = None

    indices = trainer.build_faiss_indices(
        embeddings,
        indices_to_build=indices_to_build,
        fingerprints=fingerprints
    )

    dataset_fp = compute_dataset_fingerprint(train_file)
    clf_meta = classifier_dir / "unified_tag_classifier_metadata.json"
    clf_info = compute_classifier_fingerprint(clf_meta) if clf_meta.exists() else {}
    clf_fp = clf_info.get("fingerprint")
    dependencies = {
        'dataset': {
            'fingerprint': dataset_fp,
            'num_questions': len(df),
            'file': 'question_tag.csv'
        },
    }
    if clf_fp:
        dependencies['classifier'] = {
            'fingerprint': clf_fp,
            'file': 'unified_tag_classifier.pth'
        }

    metadata = trainer.save_artifacts(
        indices, embeddings, df, models_dir, 
        fingerprint=fingerprint, 
        dependencies=dependencies,
        fp_map=fp_map
    )

    print("\n" + "=" * 80)
    print("[SUCCESS] STS TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to: {models_dir}")
    print(f"\nStatistics:")
    print(f"  Questions:        {metadata['num_questions']:,}")
    print(f"  Tags:             {metadata['num_tags']}")
    print(f"  Embedding dim:    {metadata['embedding_dim']}")
    print(f"  Indices created:  {len(indices)}")
    print(f"  Index type:       {metadata.get('index_type', 'Unknown')}")
    print(f"  Incremental:      {metadata.get('supports_incremental', False)}")
    print(f"\nIndices built:")
    for name, size in metadata['indices'].items():
        print(f"  {name:25s}: {size:,} vectors")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build FAISS indices for STS component',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all',
        action='store_true',
        help='Build the global index (default)'
    )
    group.add_argument(
        '--global',
        action='store_true',
        dest='global_only',
        help='Alias for --all (global index only)'
    )
    group.add_argument(
        '--indices',
        nargs='+',
        metavar='INDEX',
        help='Compatibility flag; only "global" is supported'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Ignore cache and rebuild artifacts'
    )

    parser.add_argument(
        '--train-csv',
        type=Path,
        default=DATASETS_DIR / "question_tag.csv",
        help='Path to training CSV'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=SEMANTIC_MODELS_DIR,
        help='Output directory for semantic models'
    )
    parser.add_argument(
        '--classifier-dir',
        type=Path,
        default=CLASSIFIER_MODELS_DIR,
        help='Directory containing the tag classifier model'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f'Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})'
    )
    parser.add_argument(
        '--normalize',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Normalize embeddings (default: True, use --no-normalize to disable)'
    )
    parser.add_argument(
        '--use-native-prompts',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use model-native prompts instead of custom prefixes'
    )
    parser.add_argument(
        '--use-prefixes',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable custom prefix formatting'
    )
    parser.add_argument(
        '--use-instruct-format',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use E5 instruct format for queries'
    )
    parser.add_argument(
        '--instruct-task',
        type=str,
        default=DEFAULT_E5_INSTRUCT_TASK,
        help='Instruction string for instruct-format queries'
    )
    parser.add_argument(
        '--sts-query-prefix',
        type=str,
        default="query: ",
        help='Prefix for STS queries when not using instruct format'
    )
    parser.add_argument(
        '--sts-passage-prefix',
        type=str,
        default="passage: ",
        help='Prefix for STS passages when not using instruct format'
    )
    parser.add_argument(
        '--classifier-query-prefix',
        type=str,
        default="query: ",
        help='Prefix for classifier queries when not using instruct format'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Directory for embedding cache (default: models_dir/../cache)'
    )
    parser.add_argument(
        '--no-embedding-cache',
        action='store_true',
        help='Disable embedding cache (encode all questions fresh)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation (not recommended)'
    )
    
    args = parser.parse_args()

    if args.global_only:
        indices_to_build = 'global'
    elif args.indices:
        indices_to_build = args.indices
    else:
        indices_to_build = 'all'

    prefixes = EmbeddingPrefixConfig(
        use_native_prompts=args.use_native_prompts,
        use_prefixes=args.use_prefixes,
        use_instruct_format=args.use_instruct_format,
        instruct_task=args.instruct_task,
        sts_query_prefix=args.sts_query_prefix,
        sts_passage_prefix=args.sts_passage_prefix,
        classifier_query_prefix=args.classifier_query_prefix,
    )

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = args.models_dir.parent / "cache"

    train_sts(
        indices_to_build=indices_to_build,
        force=args.force,
        train_file=args.train_csv,
        models_dir=args.models_dir,
        classifier_dir=args.classifier_dir,
        embedding_model=args.embedding_model,
        normalize_embeddings=args.normalize,
        prefixes=prefixes,
        cache_dir=cache_dir,
        use_embedding_cache=not args.no_embedding_cache,
        validate_data=not args.no_validate,
    )
