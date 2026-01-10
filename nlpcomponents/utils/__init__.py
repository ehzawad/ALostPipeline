from typing import TYPE_CHECKING

__all__ = [
    "get_faiss",
    "extract_ngram_words",
    "extract_all_ngrams",
    "load_tag_answers",
    "find_artifact_file",
    "format_missing_artifact_error",
    "format_validation_error",
    "PACKAGE_ROOT",
    "DATASETS_DIR",
    "FEATURES_DIR",
    "MODELS_DIR",
]

if TYPE_CHECKING:
    from .faiss_utils import get_faiss
    from .ngram_utils import extract_ngram_words, extract_all_ngrams
    from .path_utils import (
        load_tag_answers,
        find_artifact_file,
        PACKAGE_ROOT,
        DATASETS_DIR,
        FEATURES_DIR,
        MODELS_DIR,
    )
    from .errors import format_missing_artifact_error, format_validation_error

def __getattr__(name: str):
    if name == "get_faiss":
        from .faiss_utils import get_faiss
        return get_faiss

    if name in ("extract_ngram_words", "extract_all_ngrams"):
        from .ngram_utils import extract_ngram_words, extract_all_ngrams
        return extract_ngram_words if name == "extract_ngram_words" else extract_all_ngrams

    if name in ("load_tag_answers", "find_artifact_file", "PACKAGE_ROOT", "DATASETS_DIR", "FEATURES_DIR", "MODELS_DIR"):
        from .path_utils import (
            load_tag_answers,
            find_artifact_file,
            PACKAGE_ROOT,
            DATASETS_DIR,
            FEATURES_DIR,
            MODELS_DIR,
        )
        return {
            "load_tag_answers": load_tag_answers,
            "find_artifact_file": find_artifact_file,
            "PACKAGE_ROOT": PACKAGE_ROOT,
            "DATASETS_DIR": DATASETS_DIR,
            "FEATURES_DIR": FEATURES_DIR,
            "MODELS_DIR": MODELS_DIR,
        }[name]

    if name in ("format_missing_artifact_error", "format_validation_error"):
        from .errors import format_missing_artifact_error, format_validation_error
        return format_missing_artifact_error if name == "format_missing_artifact_error" else format_validation_error

    raise AttributeError(f"module 'nlpcomponents.utils' has no attribute {name!r}")
