from __future__ import annotations

from loguru import logger
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PACKAGE_ROOT / "datasets"
FEATURES_DIR = DATASETS_DIR / "features"
MODELS_DIR = PACKAGE_ROOT / "models"
SEMANTIC_MODELS_DIR = MODELS_DIR / "semantic"
CLASSIFIER_MODELS_DIR = MODELS_DIR / "tag_classifier"

def find_artifact_file(
    filename: str,
    search_paths: List[Path],
    required: bool = True
) -> Optional[Path]:
    for path in search_paths:
        full_path = path / filename if path.is_dir() else path
        if full_path.exists():
            return full_path

    if required:
        searched = ", ".join(str(p) for p in search_paths)
        raise FileNotFoundError(
            f"File '{filename}' not found. Searched: {searched}"
        )
    return None

def load_tag_answers(
    models_dir: Path,
    filename: str = "tag_answer.csv",
    required: bool = False
) -> Dict[str, str]:
    search_paths = [
        models_dir.parent / "datasets",
        models_dir,
        DATASETS_DIR,
    ]

    for search_dir in search_paths:
        path = search_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            if not {'tag', 'answer'}.issubset(df.columns):
                raise ValueError(
                    f"{filename} at {path} missing required columns 'tag' and 'answer'"
                )
            tag_to_answer = dict(zip(df['tag'], df['answer']))
            logger.info(f"  Loaded {len(tag_to_answer)} answers from {path}")
            return tag_to_answer

    searched = ", ".join(str(p / filename) for p in search_paths)
    if required:
        raise FileNotFoundError(
            f"Tag answer file '{filename}' not found. Searched: {searched}"
        )

    logger.info(f"  No tag answer file found (searched: {searched}); defaulting to empty mapping")
    return {}
