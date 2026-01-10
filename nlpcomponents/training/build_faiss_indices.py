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
from nlpcomponents.cache.model_cache import get_default_device
from nlpcomponents.build.fingerprint import compute_dataset_fingerprint, compute_classifier_fingerprint
from nlpcomponents.utils.faiss_utils import get_faiss, is_gpu_available, index_cpu_to_gpu, index_gpu_to_cpu, get_device_string
from nlpcomponents.utils.path_utils import DATASETS_DIR, SEMANTIC_MODELS_DIR, CLASSIFIER_MODELS_DIR

if TYPE_CHECKING:
    import faiss

class STSTrainer:

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        normalize_embeddings: bool = True,
        prefixes: Optional[EmbeddingPrefixConfig] = None,
    ):
        self.embedding_model_name = embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.prefixes = prefixes or EmbeddingPrefixConfig()
        
        self.embedding_model = None
        self.embedding_dim: Optional[int] = None
        self.use_unified_embeddings = False

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

    def generate_embeddings(self, questions):
        logger.info("\nGenerating embeddings...")
        logger.info(f"  Model: {self.embedding_model_name}")
        logger.info(f"  Questions: {len(questions)}")
        logger.info(f"  Normalize: {self.normalize_embeddings}")
        logger.info(f"  Native prompts: {self.prefixes.use_native_prompts}, prefixes enabled: {self.prefixes.use_prefixes}")
        
        return self._generate_raw_embeddings(questions)
    
    def _generate_raw_embeddings(self, questions):
        questions_to_encode = self.prefixes.format_sts_passages_batch(questions)

        from nlpcomponents.cache.model_cache import get_shared_embedding_model, encode_documents

        device = get_default_device()
        logger.info(f"  Device: {device}")
        if device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Version: {torch.version.cuda}")

        logger.info(f"  Loading model on {device}...")
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)
        logger.info(f"  [OK] Model loaded successfully on {device}")

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

    def build_faiss_indices(self, embeddings, indices_to_build='global', use_gpu: bool = True):
        logger.info("\nBuilding FAISS indices...")
        if self.embedding_dim is None:
            raise ValueError("embedding_dim not set; generate embeddings before building indices.")
        embeddings_f32 = embeddings.astype('float32')

        if indices_to_build not in ('all', 'global', None):
            logger.warning("Class-specific indices are no longer supported; building global index only.")

        logger.info("  Building global index...")
        logger.info(f"  FAISS device: {get_device_string()}")

        start_time = time.time()
        faiss_mod = get_faiss()
        
        index_global = faiss_mod.IndexFlatIP(self.embedding_dim)
        
        gpu_used = False
        if use_gpu and is_gpu_available():
            try:
                gpu_index, moved = index_cpu_to_gpu(index_global)
                if moved:
                    gpu_index.add(embeddings_f32)
                    index_global = index_gpu_to_cpu(gpu_index)
                    del gpu_index
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gpu_used = True
                    logger.info("  [GPU] Index built on GPU, converted to CPU for saving")
                else:
                    index_global.add(embeddings_f32)
            except Exception as e:
                logger.warning(f"  GPU index building failed, falling back to CPU: {e}")
                index_global.add(embeddings_f32)
        else:
            index_global.add(embeddings_f32)
        
        build_time = time.time() - start_time

        device_str = "GPU" if gpu_used else "CPU"
        logger.info(f"  [OK] Global index built on {device_str}: {index_global.ntotal:,} vectors in {build_time:.2f}s")
        return {'global': index_global}

    def save_artifacts(self, indices, embeddings, df, output_dir: Path, fingerprint: str | None = None, dependencies: dict | None = None):
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
            'fingerprint': fingerprint,
            'dependencies': dependencies or {},
            **self.prefixes.get_metadata()
        }

        metadata_file = output_dir / "sts_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"  [OK] Metadata: {metadata_file.name}")

        logger.info(f"\n  All artifacts saved to: {output_dir}")

        return metadata

def train_sts(
    indices_to_build='all',
    force: bool = False,
    train_file: Path = Path("nlpcomponents/datasets/sts_train.csv"),
    models_dir: Path = Path("nlpcomponents/models/semantic"),
    classifier_dir: Path = Path("nlpcomponents/models/tag_classifier"),
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    normalize_embeddings: bool = True,
    prefixes: Optional[EmbeddingPrefixConfig] = None
):
    print("=" * 80)
    print("STS TRAINING - BUILD FAISS INDICES")
    print("=" * 80)

    trainer = STSTrainer(
        embedding_model=embedding_model,
        normalize_embeddings=normalize_embeddings,
        prefixes=prefixes,
    )

    metadata_file = models_dir / "sts_metadata.json"
    cache_inputs = [train_file]
    
    cache_extra = json.dumps({
        'embedding_model': trainer.embedding_model_name,
        'normalize_embeddings': trainer.normalize_embeddings,
        'indices_to_build': indices_to_build,
        'embedding_prefix_config': trainer.prefixes.get_cache_key()
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

    embeddings = trainer.generate_embeddings(df['question'].tolist())

    indices = trainer.build_faiss_indices(
        embeddings,
        indices_to_build=indices_to_build
    )

    dataset_fp = compute_dataset_fingerprint(train_file)
    clf_meta = classifier_dir / "unified_tag_classifier_metadata.json"
    clf_info = compute_classifier_fingerprint(clf_meta) if clf_meta.exists() else {}
    clf_fp = clf_info.get("fingerprint")
    dependencies = {
        'dataset': {
            'fingerprint': dataset_fp,
            'num_questions': len(df),
            'file': 'sts_train.csv'
        },
    }
    if clf_fp:
        dependencies['classifier'] = {
            'fingerprint': clf_fp,
            'file': 'unified_tag_classifier.pth'
        }

    metadata = trainer.save_artifacts(indices, embeddings, df, models_dir, fingerprint=fingerprint, dependencies=dependencies)

    print("\n" + "=" * 80)
    print("[SUCCESS] STS TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to: {models_dir}")
    print(f"\nStatistics:")
    print(f"  Questions:        {metadata['num_questions']:,}")
    print(f"  Tags:             {metadata['num_tags']}")
    print(f"  Embedding dim:    {metadata['embedding_dim']}")
    print(f"  Indices created:  {len(indices)}")
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
        default=DATASETS_DIR / "sts_train.csv",
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

    train_sts(
        indices_to_build=indices_to_build,
        force=args.force,
        train_file=args.train_csv,
        models_dir=args.models_dir,
        classifier_dir=args.classifier_dir,
        embedding_model=args.embedding_model,
        normalize_embeddings=args.normalize,
        prefixes=prefixes,
    )
