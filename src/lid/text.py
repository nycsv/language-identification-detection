"""Gather and sanitize text input for language identification.

Both text models in this package wrap fastText, whose `predict()` rejects
newlines, so each input is collapsed to a single line.
"""

import sys
from pathlib import Path


def _clean(line: str) -> str:
    """Strip whitespace and collapse newlines into spaces."""
    return line.replace("\n", " ").replace("\r", " ").strip()


def gather_texts(texts: list[str], file: str | None) -> list[str]:
    """Collect text inputs from CLI args, a file, or stdin.

    Precedence: `--file` first, then positional `texts`, then piped stdin.
    Each input is sanitized and empty results are dropped.
    """
    if file:
        raw = Path(file).read_text(encoding="utf-8").splitlines()
    elif texts:
        raw = texts
    elif not sys.stdin.isatty():
        raw = sys.stdin.read().splitlines()
    else:
        raw = []
    return [cleaned for line in raw if (cleaned := _clean(line))]
