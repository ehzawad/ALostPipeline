import argparse
import json
from loguru import logger
import random
import time
from pathlib import Path
from datetime import datetime
import copy
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlpcomponents.cache.model_cache import get_shared_embedding_model, get_default_device, encode_queries
from nlpcomponents.build.fingerprint import compute_fingerprint
from nlpcomponents.build.fingerprint import compute_dataset_fingerprint, compute_ngram_fingerprint
from nlpcomponents.config import EmbeddingPrefixConfig, DEFAULT_E5_INSTRUCT_TASK, DEFAULT_EMBEDDING_MODEL
from nlpcomponents.inference import UnifiedTagClassifier, SupConLoss
from nlpcomponents.utils.constants import NGRAM_TYPES
from nlpcomponents.utils.ngram_utils import extract_ngram_words
from nlpcomponents.utils.errors import format_missing_artifact_error

class TagDataset(Dataset):

    def __init__(self, embeddings, patterns, labels):
        self.embeddings = embeddings
        self.patterns = patterns
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.embeddings[idx]),
            torch.FloatTensor(self.patterns[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )

class UnifiedTagClassifierTrainer:

    def __init__(
        self,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        contrastive_weight=0.1,
        features_dir: Optional[Path] = None,
        prefixes: Optional[EmbeddingPrefixConfig] = None,
        normalize_embeddings: bool = True
    ):
        self.embedding_model_name = embedding_model
        self.contrastive_weight = contrastive_weight
        self.features_dir = features_dir
        self.embedding_model = None
        self.device = None
        self.tag_encoder = LabelEncoder()
        self.prefixes = prefixes or EmbeddingPrefixConfig()
        self.normalize_embeddings = normalize_embeddings

    def load_data(self, train_file: Path, eval_file: Path):
        logger.info("Loading training and eval data...")

        df_train = pd.read_csv(train_file)
        logger.info(f"  Train: {len(df_train)} questions (tags: {df_train['tag'].nunique()})")

        df_eval = pd.read_csv(eval_file)
        logger.info(f"  Eval: {len(df_eval)} questions (tags: {df_eval['tag'].nunique()})")

        logger.info(f"\n  Top 10 tags (train):")
        tag_counts = df_train['tag'].value_counts()
        for tag, count in tag_counts.head(10).items():
            logger.info(f"    {tag:50s}: {count:4d}")

        return df_train, df_eval


    def load_ngram_features(self):
        if not self.features_dir or not self.features_dir.exists():
            raise FileNotFoundError(
                format_missing_artifact_error(
                    "Features directory",
                    self.features_dir or Path("(not set)"),
                    "python -m nlpcomponents.cli features"
                )
            )

        features_path = self.features_dir / "manual_ngrams.json"

        logger.info(f"Loading n-gram features from: {features_path}")

        if not features_path.exists():
            raise FileNotFoundError(
                format_missing_artifact_error(
                    "N-gram features file",
                    features_path,
                    "python -m nlpcomponents.cli features"
                )
            )

        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)

        tag_patterns = {}
        for tag_name, tag_data in features["tags"].items():
            tag_patterns[tag_name] = {
                "unigrams": set(item["ngram"] for item in tag_data.get("unigrams", [])),
                "bigrams": set(item["ngram"] for item in tag_data.get("bigrams", [])),
                "trigrams": set(item["ngram"] for item in tag_data.get("trigrams", [])),
                "fourgrams": set(item["ngram"] for item in tag_data.get("fourgrams", [])),
                "fivegrams": set(item["ngram"] for item in tag_data.get("fivegrams", []))
            }

        tags_sorted = sorted(tag_patterns.keys())

        logger.info(f"  Loaded features for {len(tags_sorted)} tags")
        logger.info(f"  Pattern dimension: {len(tags_sorted) * NGRAM_TYPES} ({NGRAM_TYPES} types × {len(tags_sorted)} tags)")

        for tag in tags_sorted[:3]:
            logger.info(
                f"    {tag}: {len(tag_patterns[tag]['unigrams'])} uni, "
                f"{len(tag_patterns[tag]['bigrams'])} bi, "
                f"{len(tag_patterns[tag]['trigrams'])} tri, "
                f"{len(tag_patterns[tag]['fourgrams'])} four, "
                f"{len(tag_patterns[tag]['fivegrams'])} five"
            )

        return tag_patterns, tags_sorted

    def compute_pattern_features_raw(self, questions, tag_patterns, tags_sorted):
        logger.info(f"Computing per-tag n-gram matching features ({len(tags_sorted) * NGRAM_TYPES}-dim)...")

        features = []
        for idx, question in enumerate(tqdm(questions, desc="Extracting patterns", disable=False)):
            q_unigrams = extract_ngram_words(question, 1)
            q_bigrams = extract_ngram_words(question, 2)
            q_trigrams = extract_ngram_words(question, 3)
            q_fourgrams = extract_ngram_words(question, 4)
            q_fivegrams = extract_ngram_words(question, 5)

            feature_vec = []
            for tag in tags_sorted:
                unigram_matches = len(q_unigrams & tag_patterns[tag]["unigrams"])
                bigram_matches = len(q_bigrams & tag_patterns[tag]["bigrams"])
                trigram_matches = len(q_trigrams & tag_patterns[tag]["trigrams"])
                fourgram_matches = len(q_fourgrams & tag_patterns[tag]["fourgrams"])
                fivegram_matches = len(q_fivegrams & tag_patterns[tag]["fivegrams"])
                feature_vec.extend([unigram_matches, bigram_matches, trigram_matches, fourgram_matches, fivegram_matches])

            features.append(feature_vec)

        features_array = np.array(features, dtype=np.float32)
        logger.info(f"  Pattern features shape: {features_array.shape}")

        return features_array
    
    def compute_normalization_stats(self, features_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        PATTERN_STD_EPSILON = 1e-7
        self.pattern_std_epsilon = PATTERN_STD_EPSILON
        mean = features_array.mean(axis=0, keepdims=True)
        std = features_array.std(axis=0, keepdims=True) + PATTERN_STD_EPSILON
        return mean, std
    
    def normalize_features(self, features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (features - mean) / std

    def generate_embeddings(self, questions):
        logger.info("Generating embeddings using shared model cache...")
        logger.info(f"  Native prompts: {self.prefixes.use_native_prompts}, prefixes enabled: {self.prefixes.use_prefixes}")
        if self.prefixes.use_prefixes and self.prefixes.use_instruct_format:
            logger.info(f"  Classifier query format: Instruct + Query (E5-instruct)")

        self.device = get_default_device()
        logger.info(f"  Device: {self.device}")

        logger.info(f"  Loading shared embedding model: {self.embedding_model_name}")
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)

        questions_to_encode = self.prefixes.format_classifier_queries_batch(questions)

        batch_size = 128 if self.device == 'cuda' else 32
        embeddings = encode_queries(
            self.embedding_model,
            questions_to_encode,
            use_native=self.prefixes.use_native_prompts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            batch_size=batch_size
        )

        logger.info(f"  Embeddings shape: {embeddings.shape}")
        return embeddings

    def train(self, df_train, epochs=100, batch_size=64, lr=0.001, patience=15, lr_patience=7, max_grad_norm=1.0):
        logger.info("\nTraining unified tag classifier...")
        logger.info(f"  Contrastive loss weight: {self.contrastive_weight}")
        logger.info(f"  Max epochs: {epochs}, Patience: {patience}, Max grad norm: {max_grad_norm}")

        self.hyperparameters = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'max_epochs': epochs,
            'early_stopping_patience': patience,
            'lr_patience': lr_patience,
            'max_grad_norm': max_grad_norm,
            'optimizer': 'Adam',
            'weight_decay': 1e-5,
            'warmup_epochs': 5,
            'scheduler': 'LinearLR+ReduceLROnPlateau',
            'scheduler_patience': 7,
        }

        tag_labels = self.tag_encoder.fit_transform(df_train['tag'].values)
        num_tags = len(self.tag_encoder.classes_)
        logger.info(f"  Number of tag classes: {num_tags}")

        tag_patterns, tags_sorted = self.load_ngram_features()
        self.tag_patterns = tag_patterns
        self.tags_sorted = tags_sorted

        training_tags = set(df_train['tag'].unique())
        loaded_tags = set(tags_sorted)
        missing_in_json = training_tags - loaded_tags
        if missing_in_json:
            raise ValueError(
                f"\n{len(missing_in_json)} tags in training data not found in n-gram features JSON:\n"
                f"{sorted(missing_in_json)}\n"
                f"Please regenerate features."
            )
        
        extra_in_json = loaded_tags - training_tags
        if extra_in_json:
            raise ValueError(
                f"\n{len(extra_in_json)} tags in n-gram features JSON not in training data:\n"
                f"{sorted(extra_in_json)[:10]}{'...' if len(extra_in_json) > 10 else ''}\n"
                f"This will cause dimension mismatch at inference.\n"
                f"Regenerate features with: python -m nlpcomponents.cli features --force"
            )

        embeddings = self.generate_embeddings(df_train['question'].tolist())
        
        patterns_raw = self.compute_pattern_features_raw(
            df_train['question'].tolist(), tag_patterns, tags_sorted
        )

        tag_counts = np.bincount(tag_labels)
        tags_with_single_sample = np.sum(tag_counts < 2)
        if tags_with_single_sample > 0:
            problem_tags = [tags_sorted[i] for i, count in enumerate(tag_counts) if count < 2]
            raise ValueError(
                f"Cannot perform stratified split: {tags_with_single_sample} tags have <2 samples. "
                f"Tags with insufficient samples: {problem_tags[:5]}{'...' if len(problem_tags) > 5 else ''}. "
                "Add more training data or remove these tags."
            )

        X_emb_train, X_emb_val, X_pat_train_raw, X_pat_val_raw, y_train, y_val = train_test_split(
            embeddings, patterns_raw, tag_labels,
            test_size=0.2,
            random_state=42,
            stratify=tag_labels
        )
        
        self.pattern_mean, self.pattern_std = self.compute_normalization_stats(X_pat_train_raw)
        
        X_pat_train = self.normalize_features(X_pat_train_raw, self.pattern_mean, self.pattern_std)
        X_pat_val = self.normalize_features(X_pat_val_raw, self.pattern_mean, self.pattern_std)

        logger.info(f"  Train: {len(y_train)}, Val: {len(y_val)}")

        train_dataset = TagDataset(X_emb_train, X_pat_train, y_train)
        val_dataset = TagDataset(X_emb_val, X_pat_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        embedding_dim = embeddings.shape[1]
        pattern_dim = len(tags_sorted) * NGRAM_TYPES
        logger.info(f"  Embedding dimension: {embedding_dim}")
        logger.info(f"  Pattern dimension: {pattern_dim}")

        model = UnifiedTagClassifier(
            embedding_dim=embedding_dim,
            pattern_dim=pattern_dim,
            num_tags=num_tags,
            dropout=0.5
        )
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Model parameters: {total_params:,} (trainable: {trainable_params:,})")

        ce_criterion = nn.CrossEntropyLoss()
        supcon_criterion = SupConLoss(temperature=0.1) if self.contrastive_weight > 0 else None
        
        if supcon_criterion:
            logger.info(f"  Using supervised contrastive loss (weight={self.contrastive_weight})")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        warmup_epochs = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=lr_patience
        )
        logger.info(f"  LR schedule: {warmup_epochs} epoch warmup, then ReduceLROnPlateau (patience={lr_patience})")

        logger.info("\nTraining...")
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_ce_loss = 0.0
            train_con_loss = 0.0
            train_correct = 0
            train_total = 0

            for emb, pat, labels_batch in train_loader:
                emb = emb.to(self.device)
                pat = pat.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer.zero_grad()
                
                if self.contrastive_weight > 0:
                    outputs, emb_features, fused_features = model(
                        emb, pat, return_embedding_features=True
                    )
                    ce_loss = ce_criterion(outputs, labels_batch)
                    con_loss = supcon_criterion(fused_features, labels_batch)
                    loss = ce_loss + self.contrastive_weight * con_loss
                    train_ce_loss += ce_loss.item()
                    train_con_loss += con_loss.item()
                else:
                    outputs = model(emb, pat)
                    loss = ce_criterion(outputs, labels_batch)
                    train_ce_loss += loss.item()
                
                loss.backward()
                
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()

            train_acc = 100.0 * train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for emb, pat, labels_batch in val_loader:
                    emb = emb.to(self.device)
                    pat = pat.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    outputs = model(emb, pat)
                    loss = ce_criterion(outputs, labels_batch)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels_batch.size(0)
                    val_correct += predicted.eq(labels_batch).sum().item()

            val_acc = 100.0 * val_correct / val_total

            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(val_loss)

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_ce_loss = train_ce_loss / len(train_loader)
            avg_con_loss = train_con_loss / len(train_loader) if self.contrastive_weight > 0 else 0

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                loss_detail = f"CE: {avg_ce_loss:.4f}"
                if self.contrastive_weight > 0:
                    loss_detail += f", SupCon: {avg_con_loss:.4f}"
                logger.info(f"  Epoch {epoch+1:3d}/{epochs}: "
                           f"Train Loss: {avg_train_loss:.4f} ({loss_detail}), Train Acc: {train_acc:.2f}% | "
                           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% <- NEW BEST")
            else:
                patience_counter += 1
                if (epoch + 1) % 5 == 0:
                    logger.info(f"  Epoch {epoch+1:3d}/{epochs}: "
                               f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                               f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% (patience: {patience_counter}/{patience})")

            if patience_counter >= patience:
                logger.info(f"\n  Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"  Best model was at epoch {best_epoch} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%")
                break

        self.trained_epochs = epoch + 1
        self.best_epoch = best_epoch
        self.best_val_acc = best_val_acc
        self.early_stopped = (epoch + 1) < epochs

        logger.info(f"\n  Training Summary:")
        logger.info(f"    Contrastive weight: {self.contrastive_weight}")
        logger.info(f"    Epochs trained: {self.trained_epochs}/{epochs}")
        logger.info(f"    Best epoch: {best_epoch}")
        logger.info(f"    Best val loss: {best_val_loss:.4f}")
        logger.info(f"    Best val accuracy: {best_val_acc:.2f}%")
        if self.early_stopped:
            logger.info(f"    Early stopped: Yes (patience reached)")
        else:
            logger.info(f"    Early stopped: No (max epochs reached)")

        model.load_state_dict(self.best_model_state)
        return model

    def evaluate(self, model, df_eval):
        logger.info("\nEvaluating on eval set...")

        tag_labels = self.tag_encoder.transform(df_eval['tag'].values)

        logger.info("Generating embeddings (using cached model)...")
        questions_to_encode = self.prefixes.format_classifier_queries_batch(df_eval['question'].tolist())
        
        batch_size = 128 if self.device == 'cuda' else 32
        embeddings = encode_queries(
            self.embedding_model,
            questions_to_encode,
            use_native=self.prefixes.use_native_prompts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            batch_size=batch_size
        )
        logger.info(f"  Embeddings shape: {embeddings.shape}")

        patterns_raw = self.compute_pattern_features_raw(
            df_eval['question'].tolist(),
            self.tag_patterns,
            self.tags_sorted,
        )
        patterns = self.normalize_features(patterns_raw, self.pattern_mean, self.pattern_std)

        eval_dataset = TagDataset(embeddings, patterns, tag_labels)
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for emb, pat, labels_batch in eval_loader:
                emb = emb.to(self.device)
                pat = pat.to(self.device)
                labels_batch = labels_batch.to(self.device)

                outputs = model(emb, pat)
                _, predicted = outputs.max(1)

                total += labels_batch.size(0)
                correct += predicted.eq(labels_batch).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        overall_accuracy = 100.0 * correct / total
        logger.info(f"\n  Overall eval accuracy: {overall_accuracy:.2f}% ({correct}/{total})")

        return overall_accuracy, all_predictions, all_labels, {}

    def save_model(self, model, output_dir, fingerprint: str | None = None, dependencies: dict | None = None):
        logger.info("\nSaving unified model...")

        output_dir.mkdir(parents=True, exist_ok=True)

        model_file = output_dir / "unified_tag_classifier.pth"
        
        model_config = model.get_config() if hasattr(model, 'get_config') else {}
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'tag_encoder_classes': self.tag_encoder.classes_.tolist(),
            'pattern_mean': self.pattern_mean.tolist(),
            'pattern_std': self.pattern_std.tolist(),
            'pattern_std_epsilon': getattr(self, 'pattern_std_epsilon', 1e-7),
            'embedding_model': self.embedding_model_name,
            'num_tags': len(self.tag_encoder.classes_),
            'contrastive_weight': self.contrastive_weight,
            'trained_epochs': self.trained_epochs,
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'early_stopped': self.early_stopped,
            'pattern_dim': model_config.get('pattern_dim', len(self.tags_sorted) * NGRAM_TYPES),
            'embedding_dim': model_config.get('embedding_dim', 1024),
            'dropout': model_config.get('dropout', 0.5),
            'hyperparameters': getattr(self, 'hyperparameters', {}),
            'normalize_embeddings': self.normalize_embeddings,
            **self.prefixes.get_metadata()
        }, model_file)

        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"  Model saved: {model_file.name} ({file_size_mb:.2f} MB)")

        metadata = {
            'contrastive_weight': self.contrastive_weight,
            'embedding_model': self.embedding_model_name,
            'num_tags': len(self.tag_encoder.classes_),
            'tags': self.tag_encoder.classes_.tolist(),
            'trained_epochs': self.trained_epochs,
            'best_epoch': self.best_epoch,
            'best_val_acc': float(self.best_val_acc),
            'early_stopped': self.early_stopped,
            'created_at': datetime.now().isoformat(),
            'fingerprint': fingerprint,
            'dependencies': dependencies or {},
            'normalize_embeddings': self.normalize_embeddings,
            **self.prefixes.get_metadata()
        }

        metadata_file = output_dir / "unified_tag_classifier_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"  Metadata saved: {metadata_file.name}")
        logger.info(f"\n  All artifacts saved to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train unified tag classifier with attention and optional contrastive loss"
    )
    parser.add_argument(
        "--models",
        default="nlpcomponents/models/tag_classifier",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer backbone"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs (trainer uses early stopping)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience"
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.1,
        help="Weight for supervised contrastive loss (0 to disable, default: 0.1)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and retrain"
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("nlpcomponents/datasets/sts_train.csv"),
        help="Path to training CSV"
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("nlpcomponents/datasets/sts_eval.csv"),
        help="Path to eval CSV"
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("nlpcomponents/datasets/features"),
        help="Directory containing n-gram feature files"
    )
    parser.add_argument(
        "--use-native-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use model-native prompts instead of custom prefixes"
    )
    parser.add_argument(
        "--use-prefixes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable custom prefix formatting"
    )
    parser.add_argument(
        "--use-instruct-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use E5 instruct format for queries"
    )
    parser.add_argument(
        "--instruct-task",
        type=str,
        default=DEFAULT_E5_INSTRUCT_TASK,
        help="Instruction string for instruct-format queries"
    )
    parser.add_argument(
        "--sts-query-prefix",
        type=str,
        default="query: ",
        help="Prefix for STS queries when not using instruct format"
    )
    parser.add_argument(
        "--sts-passage-prefix",
        type=str,
        default="passage: ",
        help="Prefix for STS passages when not using instruct format"
    )
    parser.add_argument(
        "--classifier-query-prefix",
        type=str,
        default="query: ",
        help="Prefix for classifier queries when not using instruct format"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 80)
    print("UNIFIED TAG CLASSIFIER TRAINING")
    print("=" * 80)

    ROOT_DIR = Path(__file__).resolve().parents[2]
    models_dir = Path(args.models)
    if not models_dir.is_absolute():
        models_dir = ROOT_DIR / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {models_dir.resolve()}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Using manual n-gram features")
    print(f"Contrastive weight: {args.contrastive_weight}")
    print()

    try:
        logger.info("Initializing trainer...")
        prefixes = EmbeddingPrefixConfig(
            use_native_prompts=args.use_native_prompts,
            use_prefixes=args.use_prefixes,
            use_instruct_format=args.use_instruct_format,
            instruct_task=args.instruct_task,
            sts_query_prefix=args.sts_query_prefix,
            sts_passage_prefix=args.sts_passage_prefix,
            classifier_query_prefix=args.classifier_query_prefix,
        )

        trainer = UnifiedTagClassifierTrainer(
            embedding_model=args.embedding_model,
            contrastive_weight=args.contrastive_weight,
            features_dir=args.features_dir,
            prefixes=prefixes,
        )
        
        df_train, df_eval = trainer.load_data(args.train_csv, args.eval_csv)

        num_unique_tags = df_train['tag'].nunique()
        pattern_dim = num_unique_tags * NGRAM_TYPES

        logger.info(f"\n  Dataset summary:")
        logger.info(f"    Training questions: {len(df_train)}")
        logger.info(f"    Eval questions: {len(df_eval)}")
        logger.info(f"    Unique tags: {num_unique_tags}")

        print(f"\nThis trains a unified model for all {num_unique_tags} tags.")
        print(f"Architecture: Embedding + Pattern + Attention ({pattern_dim}-dim patterns) -> Tag prediction")
        if args.contrastive_weight > 0:
            print(f"  Loss: CrossEntropy + {args.contrastive_weight} * SupervisedContrastive")
        print()

        features_file = args.features_dir / "manual_ngrams.json"
        metadata_file = models_dir / "unified_tag_classifier_metadata.json"
        model_file = models_dir / "unified_tag_classifier.pth"

        cache_inputs = [
            args.train_csv,
            args.eval_csv,
            features_file,
        ]
        cache_extra = json.dumps({
            'embedding_model': args.embedding_model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'patience': args.patience,
            'contrastive_weight': args.contrastive_weight,
            'embedding_prefix_config': trainer.prefixes.get_cache_key()
        }, sort_keys=True)
        fingerprint = compute_fingerprint(cache_inputs, cache_extra)

        if not args.force and metadata_file.exists() and model_file.exists():
            try:
                with metadata_file.open('r', encoding='utf-8') as handle:
                    existing = json.load(handle)
            except json.JSONDecodeError:
                existing = {}
            if existing.get('fingerprint') == fingerprint:
                logger.info("\n  No input changes detected; reusing cached classifier.")
                logger.info("  Use --force to retrain.")
                print("\n" + "=" * 80)
                print("TRAINING SKIPPED - Using cached model")
                print("=" * 80)
                print(f"\nCached model: {model_file}")
                print(f"Training was completed at: {existing.get('created_at', 'unknown')}")
                cached_acc = existing.get('best_val_acc')
                if isinstance(cached_acc, (int, float)):
                    acc_str = f"{cached_acc:.2f}%"
                else:
                    acc_str = str(cached_acc or "unknown")
                print(f"Best val accuracy: {acc_str}")
                print(f"\nTo retrain, run: python {Path(__file__).name} --force")
                print("=" * 80)
                return

        logger.info(f"Training rows: {len(df_train)} | Eval rows: {len(df_eval)} | Tags: {df_train['tag'].nunique()}")
        start_time = time.time()
        model = trainer.train(
            df_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience
        )
        train_time = time.time() - start_time
        training_minutes = train_time / 60

        dataset_fp = compute_dataset_fingerprint(args.train_csv)
        eval_fp = compute_dataset_fingerprint(args.eval_csv)
        ngram_fp = compute_ngram_fingerprint(features_file)
        dependencies = {
            'dataset': {
                'fingerprint': dataset_fp,
                'num_tags': df_train['tag'].nunique(),
                'file': args.train_csv.name
            },
            'eval_dataset': {
                'fingerprint': eval_fp,
                'file': args.eval_csv.name
            },
            'ngrams': {
                'fingerprint': ngram_fp,
                'file': str(features_file.name)
            }
        }

        trainer.save_model(model, models_dir, fingerprint=fingerprint, dependencies=dependencies)

        logger.info(f"Training finished in {training_minutes:.2f} minutes")
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nModel: unified_tag_classifier.pth")
        print(f"Training time: {train_time:.1f}s ({training_minutes:.1f} minutes)")
        print(f"\nTraining Details:")
        print(f"  Epochs trained: {trainer.trained_epochs}/{args.epochs}")
        print(f"  Best epoch: {trainer.best_epoch}")
        print(f"  Best val accuracy: {trainer.best_val_acc:.2f}%")
        print(f"  Early stopped: {'Yes' if trainer.early_stopped else 'No'}")
        if args.contrastive_weight > 0:
            print(f"  Contrastive loss: Enabled (weight={args.contrastive_weight})")
        print(f"\nModel saved to: {models_dir / 'unified_tag_classifier.pth'}")
        print(f"\nTo evaluate:")
        print(f"   python -m nlpcomponents.cli eval --data nlpcomponents/datasets/sts_eval.csv --top-k 1")
        print("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
