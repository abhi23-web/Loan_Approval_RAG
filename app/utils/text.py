"""Text helpers shared by extraction, chunking and prompt assembly."""

from __future__ import annotations

import re
from datetime import date

# Sentence boundary that tolerates the abbreviations and clause numbers common in
# policy documents ("Clause 4.2 applies." must not split at "4.").
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'‘“])")

_WHITESPACE_RUN = re.compile(r"[ \t ]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")

_DECLARED_VERSION_PATTERNS = (
    re.compile(r"document\s+version\s*[:\-]?\s*([\w.\-]+)", re.IGNORECASE),
    re.compile(r"\bversion\s*[:\-]\s*([\w.\-]+)", re.IGNORECASE),
    re.compile(r"\bv(?:er)?\.?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
)

_DECLARED_DATE_PATTERNS = (
    re.compile(r"effective\s+(?:date|from)\s*[:\-]?\s*(\d{4})-(\d{2})-(\d{2})", re.IGNORECASE),
    re.compile(
        r"effective\s+(?:date|from)\s*[:\-]?\s*(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})",
        re.IGNORECASE,
    ),
)

_MONTH_NUMBER_BY_NAME = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines while preserving paragraph breaks.

    Paragraph breaks survive because the recursive chunker splits on them first;
    flattening them would push it straight down to character-level splitting.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    collapsed = "\n".join(line.strip() for line in collapsed.split("\n"))
    return _BLANK_LINE_RUN.sub("\n\n", collapsed).strip()


def split_into_sentences(text: str) -> list[str]:
    """Split into sentences for the semantic chunker."""
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def extract_declared_version(text: str) -> str | None:
    """Return the version the document claims about itself, if it states one.

    Preferred over the registry's hand-maintained ``version`` field, which drifts.
    """
    header = text[:4000]  # version statements live in the header, not clause 47
    for pattern in _DECLARED_VERSION_PATTERNS:
        match = pattern.search(header)
        if match:
            return match.group(1)
    return None


def extract_declared_effective_date(text: str) -> date | None:
    """Return the effective date stated in the document header, if any."""
    header = text[:4000]
    for pattern in _DECLARED_DATE_PATTERNS:
        match = pattern.search(header)
        if not match:
            continue
        groups = match.groups()
        try:
            if len(groups) == 3 and groups[1].isalpha():
                day_text, month_name, year_text = groups
                return date(int(year_text), _MONTH_NUMBER_BY_NAME[month_name.lower()], int(day_text))
            year_text, month_text, day_text = groups
            return date(int(year_text), int(month_text), int(day_text))
        except (ValueError, KeyError):
            continue
    return None


def truncate_for_display(text: str, max_characters: int = 320) -> str:
    """Shorten an excerpt for the UI without cutting mid-word."""
    if len(text) <= max_characters:
        return text
    cut = text[:max_characters].rsplit(" ", 1)[0]
    return f"{cut}…"


def estimate_token_count(text: str) -> int:
    """Rough token estimate used for cost/context reporting.

    Deliberately a heuristic: the exact tokeniser differs per model, and this
    number is only used for comparing configurations against each other, where a
    consistent approximation is enough. Exact counts come from the model's own
    usage fields when it reports them.
    """
    return max(1, round(len(text) / 4))
