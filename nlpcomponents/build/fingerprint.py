from __future__ import annotations

import hashlib
import json
from loguru import logger
from pathlib import Path
from typing import Dict, Optional, Tuple

def compute_dataset_fingerprint(csv_path: Path, columns: Tuple[str, ...] = ('question', 'tag')) -> str:
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        logger.warning(f"Dataset fingerprint: file not found at {csv_path}")
        return "missing"
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        logger.debug(f"Dataset fingerprint: loaded {len(df)} rows from {csv_path.name}")
    except Exception as e:
        logger.error(f"Dataset fingerprint: failed to read {csv_path}: {e}")
        return "unreadable"
    
    available_cols = [c for c in columns if c in df.columns]
    if not available_cols:
        logger.warning(f"Dataset fingerprint: no matching columns in {csv_path.name}. "
                   f"Requested: {columns}, Available: {list(df.columns)}")
        return "no_matching_columns"
    
    df_filtered = df[available_cols]
    num_rows = len(df_filtered)
    
    content_hash = hashlib.sha256(
        df_filtered.to_csv(index=False).encode()
    ).hexdigest()[:32]
    
    fingerprint_data = {
        'num_rows': num_rows,
        'columns': available_cols,
        'content_hash': content_hash,
    }
    
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    logger.info(f"Dataset fingerprint computed: {csv_path.name} -> {fingerprint[:12]}... "
             f"({num_rows} rows, {len(available_cols)} columns)")
    
    return fingerprint

def compute_file_fingerprint(file_path: Path) -> str:
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File fingerprint: file not found at {file_path}")
        return "missing"
    
    try:
        content = file_path.read_bytes()
        fingerprint = hashlib.sha256(content).hexdigest()[:32]
        logger.debug(f"File fingerprint: {file_path.name} -> {fingerprint[:12]}... "
                 f"({len(content)} bytes)")
        return fingerprint
    except Exception as e:
        logger.error(f"File fingerprint: failed to read {file_path}: {e}")
        return "unreadable"

def compute_fingerprint(
    file_paths: list,
    extra: str = ""
) -> str:
    fingerprints = []
    
    for path in file_paths:
        path = Path(path)
        fp = compute_file_fingerprint(path)
        fingerprints.append(f"{path.name}:{fp}")
    
    combined = "\n".join(fingerprints)
    if extra:
        combined += f"\n__extra__:{extra}"
    
    final_fingerprint = hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    logger.debug(f"Combined fingerprint: {len(file_paths)} files -> {final_fingerprint[:12]}...")
    
    return final_fingerprint

def compute_ngram_fingerprint(json_path: Path) -> str:
    json_path = Path(json_path)
    
    if not json_path.exists():
        logger.warning(f"N-gram fingerprint: file not found at {json_path}")
        return "missing"
    
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        num_tags = metadata.get('num_tags', 0)
        top_k = metadata.get('top_k', 0)
        cleaned = metadata.get('cleaned', False)
        use_tfidf = metadata.get('use_tfidf', False)
        
        tags_data = data.get('tags', {})
        content_parts = []
        for tag in sorted(tags_data.keys()):
            tag_ngrams = tags_data[tag]
            for ngram_type in ['unigrams', 'bigrams', 'trigrams', 'fourgrams', 'fivegrams']:
                ngrams = tag_ngrams.get(ngram_type, [])
                ngram_strs = sorted(
                    item.get('ngram', item) if isinstance(item, dict) else item
                    for item in ngrams
                )
                content_parts.append(f"{tag}:{ngram_type}:{','.join(ngram_strs)}")
        
        content_hash = hashlib.sha256('\n'.join(content_parts).encode()).hexdigest()[:32]
        
        fingerprint_data = {
            'num_tags': num_tags,
            'top_k': top_k,
            'cleaned': cleaned,
            'use_tfidf': use_tfidf,
            'content_hash': content_hash
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        
        logger.info(f"N-gram fingerprint computed: {json_path.name} -> {fingerprint[:12]}... "
                f"({num_tags} tags, top_k={top_k}, cleaned={cleaned}, tfidf={use_tfidf})")
        
        return fingerprint
    except json.JSONDecodeError as e:
        logger.error(f"N-gram fingerprint: invalid JSON in {json_path}: {e}")
        return "invalid_json"
    except Exception as e:
        logger.error(f"N-gram fingerprint: failed to read {json_path}: {e}")
        return "unreadable"

def compute_classifier_fingerprint(metadata_path: Path) -> Dict:
    metadata_path = Path(metadata_path)
    
    result = {
        'fingerprint': None,
        'dependencies': {},
        'valid': False
    }
    
    if not metadata_path.exists():
        logger.warning(f"Classifier fingerprint: metadata not found at {metadata_path}")
        return result
    
    try:
        with metadata_path.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        result['fingerprint'] = metadata.get('fingerprint')
        result['dependencies'] = metadata.get('dependencies', {})
        result['num_tags'] = metadata.get('num_tags')
        result['best_val_acc'] = metadata.get('best_val_acc')
        result['valid'] = result['fingerprint'] is not None
        
        val_acc_str = f"{result['best_val_acc']:.2f}%" if result['best_val_acc'] else 'N/A'
        fp_short = result['fingerprint'][:12] if result['fingerprint'] else 'None'
        logger.info(
            f"Classifier fingerprint extracted: {fp_short}... "
            f"({result['num_tags']} tags, val_acc={val_acc_str})"
        )
        
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Classifier fingerprint: invalid JSON in {metadata_path}: {e}")
        return result
    except Exception as e:
        logger.error(f"Classifier fingerprint: failed to read {metadata_path}: {e}")
        return result

def compute_faiss_fingerprint(metadata_path: Path) -> Dict:
    metadata_path = Path(metadata_path)
    
    result = {
        'fingerprint': None,
        'dependencies': {},
        'valid': False
    }
    
    if not metadata_path.exists():
        logger.warning(f"FAISS fingerprint: metadata not found at {metadata_path}")
        return result
    
    try:
        with metadata_path.open('r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        result['fingerprint'] = metadata.get('fingerprint')
        result['dependencies'] = metadata.get('dependencies', {})
        result['num_questions'] = metadata.get('num_questions')
        result['valid'] = result['fingerprint'] is not None
        
        logger.info(f"FAISS fingerprint extracted: {result['fingerprint'][:12] if result['fingerprint'] else 'None'}... "
                f"({result['num_questions']} questions)")
        
        return result
    except json.JSONDecodeError as e:
        logger.error(f"FAISS fingerprint: invalid JSON in {metadata_path}: {e}")
        return result
    except Exception as e:
        logger.error(f"FAISS fingerprint: failed to read {metadata_path}: {e}")
        return result

def validate_artifact_consistency(
    datasets_dir: Path,
    features_dir: Path,
    classifier_dir: Path,
    semantic_dir: Path
) -> Dict[str, Dict]:
    datasets_dir = Path(datasets_dir)
    features_dir = Path(features_dir)
    classifier_dir = Path(classifier_dir)
    semantic_dir = Path(semantic_dir)
    
    logger.info(f"Validating artifact consistency...")
    logger.info(f"  Datasets: {datasets_dir}")
    logger.info(f"  Features: {features_dir}")
    logger.info(f"  Classifier: {classifier_dir}")
    logger.info(f"  Semantic: {semantic_dir}")
    
    results = {}
    
    train_csv = datasets_dir / "question_tag.csv"
    eval_csv = datasets_dir / "eval.csv"
    
    try:
        current_train_fp = compute_dataset_fingerprint(train_csv)
        current_eval_fp = compute_dataset_fingerprint(eval_csv)
    except Exception as e:
        logger.error(f"Failed to compute dataset fingerprints: {e}")
        raise
    
    results['question_tag.csv'] = {
        'status': 'present' if current_train_fp not in ('missing', 'unreadable', 'no_matching_columns') else current_train_fp,
        'fingerprint': current_train_fp
    }
    results['eval.csv'] = {
        'status': 'present' if current_eval_fp not in ('missing', 'unreadable', 'no_matching_columns') else current_eval_fp,
        'fingerprint': current_eval_fp
    }
    
    ngram_path = features_dir / "manual_ngrams.json"
    try:
        current_ngram_fp = compute_ngram_fingerprint(ngram_path)
    except Exception as e:
        logger.error(f"Failed to compute n-gram fingerprint: {e}")
        current_ngram_fp = "error"
    
    if current_ngram_fp in ('missing', 'unreadable', 'invalid_json', 'error'):
        results['manual_ngrams.json'] = {'status': current_ngram_fp, 'fingerprint': None}
    else:
        try:
            with ngram_path.open('r', encoding='utf-8') as f:
                ngram_data = json.load(f)
            deps = ngram_data.get('metadata', {}).get('dependencies', {})
            stored_dataset_fp = deps.get('dataset', {}).get('fingerprint')
            
            if stored_dataset_fp and stored_dataset_fp != current_train_fp:
                results['manual_ngrams.json'] = {
                    'status': 'stale',
                    'fingerprint': current_ngram_fp,
                    'reason': f'dataset changed (stored: {stored_dataset_fp[:12]}..., current: {current_train_fp[:12]}...)'
                }
                logger.warning(f"N-gram features are stale: dataset fingerprint mismatch")
            else:
                results['manual_ngrams.json'] = {
                    'status': 'up-to-date',
                    'fingerprint': current_ngram_fp
                }
        except Exception as e:
            logger.warning(f"Could not verify n-gram dependencies: {e}")
            results['manual_ngrams.json'] = {
                'status': 'up-to-date',
                'fingerprint': current_ngram_fp
            }
    
    classifier_meta = classifier_dir / "unified_tag_classifier_metadata.json"
    classifier_model = classifier_dir / "unified_tag_classifier.pth"
    
    if not classifier_model.exists():
        results['unified_tag_classifier.pth'] = {'status': 'missing', 'fingerprint': None}
        logger.warning(f"Classifier model not found at {classifier_model}")
    else:
        try:
            clf_info = compute_classifier_fingerprint(classifier_meta)
        except Exception as e:
            logger.error(f"Failed to compute classifier fingerprint: {e}")
            clf_info = {'valid': False}
        
        if not clf_info['valid']:
            results['unified_tag_classifier.pth'] = {
                'status': 'unknown',
                'fingerprint': None,
                'reason': 'metadata missing or invalid'
            }
        else:
            deps = clf_info.get('dependencies', {})
            stored_dataset_fp = deps.get('dataset', {}).get('fingerprint')
            stored_ngram_fp = deps.get('ngrams', {}).get('fingerprint')
            
            is_stale = False
            reasons = []
            
            if stored_dataset_fp and stored_dataset_fp != current_train_fp:
                is_stale = True
                reasons.append(f'dataset changed ({stored_dataset_fp[:12]}... -> {current_train_fp[:12]}...)')
            
            if stored_ngram_fp and stored_ngram_fp != current_ngram_fp:
                is_stale = True
                reasons.append(f'ngrams changed ({stored_ngram_fp[:12]}... -> {current_ngram_fp[:12]}...)')
            
            if is_stale:
                logger.warning(f"Classifier is stale: {', '.join(reasons)}")
            
            results['unified_tag_classifier.pth'] = {
                'status': 'stale' if is_stale else 'up-to-date',
                'fingerprint': clf_info['fingerprint'],
                'reason': ', '.join(reasons) if reasons else None
            }
    
    faiss_meta = semantic_dir / "sts_metadata.json"
    faiss_index = semantic_dir / "faiss_index_global.index"
    
    if not faiss_index.exists():
        results['faiss_index_global.index'] = {'status': 'missing', 'fingerprint': None}
        logger.warning(f"FAISS index not found at {faiss_index}")
    else:
        try:
            faiss_info = compute_faiss_fingerprint(faiss_meta)
        except Exception as e:
            logger.error(f"Failed to compute FAISS fingerprint: {e}")
            faiss_info = {'valid': False}
        
        if not faiss_info['valid']:
            results['faiss_index_global.index'] = {
                'status': 'unknown',
                'fingerprint': None,
                'reason': 'metadata missing or invalid'
            }
        else:
            deps = faiss_info.get('dependencies', {})
            stored_dataset_fp = deps.get('dataset', {}).get('fingerprint')
            
            is_stale = False
            reasons = []
            
            if stored_dataset_fp and stored_dataset_fp != current_train_fp:
                is_stale = True
                reasons.append(f'dataset changed ({stored_dataset_fp[:12]}... -> {current_train_fp[:12]}...)')
            
            if is_stale:
                logger.warning(f"FAISS index is stale: {', '.join(reasons)}")
            
            results['faiss_index_global.index'] = {
                'status': 'stale' if is_stale else 'up-to-date',
                'fingerprint': faiss_info['fingerprint'],
                'reason': ', '.join(reasons) if reasons else None
            }
    
    statuses = [v.get('status') for v in results.values()]
    num_stale = statuses.count('stale')
    num_missing = statuses.count('missing')
    num_ok = statuses.count('up-to-date') + statuses.count('present')
    
    logger.info(f"Artifact consistency check complete: "
            f"{num_ok} up-to-date, {num_stale} stale, {num_missing} missing")
    
    return results
