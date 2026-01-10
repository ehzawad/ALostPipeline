from __future__ import annotations

from pathlib import Path

def format_missing_artifact_error(
    artifact_name: str,
    path: Path,
    fix_command: str,
    additional_info: str = ""
) -> str:
    lines = [
        "",
        "=" * 80,
        f"[ERROR] {artifact_name} not found: {path}",
        "=" * 80,
        "",
        "To fix this issue, run:",
        "",
        f"  {fix_command}",
        "",
    ]

    if additional_info:
        lines.extend([additional_info, ""])

    lines.append("=" * 80)

    return "\n".join(lines)

def format_validation_error(
    context: str,
    details: str,
    fix_suggestion: str = ""
) -> str:
    lines = [
        "",
        "=" * 80,
        f"[VALIDATION ERROR] {context}",
        "=" * 80,
        "",
        details,
        "",
    ]

    if fix_suggestion:
        lines.extend([
            "Suggested fix:",
            "",
            f"  {fix_suggestion}",
            "",
        ])

    lines.append("=" * 80)

    return "\n".join(lines)
