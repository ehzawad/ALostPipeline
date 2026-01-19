"""
Data validation module for detecting data quality issues before build.

This module provides pre-build validation that:
- Normalizes all questions using the standard text normalizer
- Detects duplicate normalized questions across different tags (data pollution)
- Fails the build with a detailed report if duplicates are found
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from ..preprocessing.normalizer import TextNormalizer, DEFAULT_NORMALIZER


@dataclass
class DuplicateEntry:
    """Represents a duplicate question found across multiple tags."""
    normalized_question: str
    original_questions: List[str]
    tags: List[str]
    row_indices: List[int]
    
    def __str__(self) -> str:
        tags_str = ", ".join(f"'{t}'" for t in self.tags)
        rows_str = ", ".join(str(r) for r in self.row_indices)
        # Truncate long questions for display
        display_q = self.normalized_question[:80] + "..." if len(self.normalized_question) > 80 else self.normalized_question
        return f'"{display_q}" -> [{tags_str}] (rows {rows_str})'


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    total_questions: int
    unique_questions: int
    duplicates: List[DuplicateEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)
    
    @property
    def affected_rows(self) -> int:
        """Total number of rows affected by duplicates."""
        return sum(len(d.row_indices) for d in self.duplicates)
    
    def get_report(self) -> str:
        """Generate a human-readable validation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("DATA VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"\nTotal questions: {self.total_questions}")
        lines.append(f"Unique normalized questions: {self.unique_questions}")
        
        if self.valid:
            lines.append("\n[OK] No data pollution detected.")
        else:
            lines.append(f"\n[ERROR] Data pollution detected - {self.duplicate_count} questions appear under multiple tags:")
            lines.append("")
            for i, dup in enumerate(self.duplicates[:20], 1):  # Show first 20
                lines.append(f"  {i}. {dup}")
            if len(self.duplicates) > 20:
                lines.append(f"\n  ... and {len(self.duplicates) - 20} more duplicates")
            lines.append(f"\nTotal affected rows: {self.affected_rows}")
            lines.append("\nFix these duplicates before rebuilding.")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


class DataValidator:
    """
    Validates question_tag.csv data for quality issues before build.
    
    Primary validation: Detects duplicate questions across different tags.
    A question should only belong to ONE tag - duplicates indicate data pollution.
    """
    
    def __init__(
        self,
        normalizer: Optional[TextNormalizer] = None,
        question_col: str = "question",
        tag_col: str = "tag",
    ):
        self.normalizer = normalizer or DEFAULT_NORMALIZER
        self.question_col = question_col
        self.tag_col = tag_col
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame for data quality issues.
        
        Args:
            df: DataFrame with question and tag columns
            
        Returns:
            ValidationResult with validation status and any issues found
        """
        logger.info("Validating data quality...")
        
        if self.question_col not in df.columns:
            raise ValueError(f"Missing required column: '{self.question_col}'")
        if self.tag_col not in df.columns:
            raise ValueError(f"Missing required column: '{self.tag_col}'")
        
        total_questions = len(df)
        logger.info(f"  Total rows: {total_questions}")
        
        # Build mapping of normalized question -> list of (original, tag, row_index)
        normalized_map: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        
        for idx, row in df.iterrows():
            question = str(row[self.question_col]) if pd.notna(row[self.question_col]) else ""
            tag = str(row[self.tag_col]) if pd.notna(row[self.tag_col]) else ""
            
            if not question.strip():
                continue
                
            normalized = self.normalizer.normalize(question)
            normalized_map[normalized].append((question, tag, int(idx)))
        
        unique_questions = len(normalized_map)
        logger.info(f"  Unique normalized questions: {unique_questions}")
        
        # Find duplicates (same question under different tags)
        duplicates: List[DuplicateEntry] = []
        
        for normalized, entries in normalized_map.items():
            # Get unique tags for this question
            tags_seen: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
            for original, tag, row_idx in entries:
                tags_seen[tag].append((original, row_idx))
            
            # If question appears under multiple different tags, it's a duplicate
            if len(tags_seen) > 1:
                all_originals = []
                all_tags = []
                all_rows = []
                
                for tag, originals_and_rows in sorted(tags_seen.items()):
                    for original, row_idx in originals_and_rows:
                        all_originals.append(original)
                        all_tags.append(tag)
                        all_rows.append(row_idx)
                
                duplicates.append(DuplicateEntry(
                    normalized_question=normalized,
                    original_questions=all_originals,
                    tags=list(tags_seen.keys()),
                    row_indices=all_rows
                ))
        
        # Sort duplicates by number of affected rows (most problematic first)
        duplicates.sort(key=lambda d: len(d.row_indices), reverse=True)
        
        valid = len(duplicates) == 0
        
        if duplicates:
            logger.error(f"  Data pollution: {len(duplicates)} questions appear under multiple tags")
        else:
            logger.info("  [OK] No duplicate questions across tags")
        
        # Check for other warnings
        warnings = []
        
        # Check for empty questions
        empty_count = df[self.question_col].isna().sum() + (df[self.question_col] == "").sum()
        if empty_count > 0:
            warnings.append(f"{empty_count} empty questions found")
        
        # Check for empty tags
        empty_tags = df[self.tag_col].isna().sum() + (df[self.tag_col] == "").sum()
        if empty_tags > 0:
            warnings.append(f"{empty_tags} empty tags found")
        
        return ValidationResult(
            valid=valid,
            total_questions=total_questions,
            unique_questions=unique_questions,
            duplicates=duplicates,
            warnings=warnings
        )
    
    def validate_file(self, csv_path: Path) -> ValidationResult:
        """
        Validate a CSV file for data quality issues.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            ValidationResult with validation status and any issues found
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading {csv_path.name} for validation...")
        df = pd.read_csv(csv_path)
        
        return self.validate_dataframe(df)
    
    def validate_and_fail_on_duplicates(
        self,
        csv_path: Path,
        raise_on_duplicates: bool = True
    ) -> ValidationResult:
        """
        Validate data and optionally raise an exception if duplicates are found.
        
        Args:
            csv_path: Path to the CSV file
            raise_on_duplicates: If True, raise DataPollutionError on duplicates
            
        Returns:
            ValidationResult
            
        Raises:
            DataPollutionError: If duplicates found and raise_on_duplicates is True
        """
        result = self.validate_file(csv_path)
        
        if not result.valid and raise_on_duplicates:
            raise DataPollutionError(result)
        
        return result


class DataPollutionError(Exception):
    """Raised when data pollution (duplicate questions across tags) is detected."""
    
    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        super().__init__(self._build_message())
    
    def _build_message(self) -> str:
        result = self.validation_result
        lines = [
            f"Data pollution detected: {result.duplicate_count} questions appear under multiple tags.",
            f"Affected rows: {result.affected_rows}",
            "",
            "First 5 duplicates:"
        ]
        for dup in result.duplicates[:5]:
            lines.append(f"  - {dup}")
        if result.duplicate_count > 5:
            lines.append(f"  ... and {result.duplicate_count - 5} more")
        lines.append("")
        lines.append("Fix these duplicates in question_tag.csv before rebuilding.")
        return "\n".join(lines)


def validate_dataset(
    csv_path: Path,
    normalizer: Optional[TextNormalizer] = None,
    fail_on_duplicates: bool = True
) -> ValidationResult:
    """
    Convenience function to validate a dataset.
    
    Args:
        csv_path: Path to the CSV file
        normalizer: Optional custom normalizer
        fail_on_duplicates: If True, raise exception on duplicates
        
    Returns:
        ValidationResult
        
    Raises:
        DataPollutionError: If duplicates found and fail_on_duplicates is True
    """
    validator = DataValidator(normalizer=normalizer)
    return validator.validate_and_fail_on_duplicates(
        csv_path,
        raise_on_duplicates=fail_on_duplicates
    )
