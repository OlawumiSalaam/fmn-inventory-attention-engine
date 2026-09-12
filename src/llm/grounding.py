"""Validation helpers for numerical grounding of generated answers."""

from __future__ import annotations

import math
import re
from typing import Any

SKU_ID = re.compile(r"\bSKU\W?\d+", re.IGNORECASE)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
WRITTEN_DATE = re.compile(
    r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)(\s+\d{4})?\b",
    re.IGNORECASE,
)
SPACED_THOUSANDS = re.compile(r"(?<=\d)[ \u00a0\u2009\u202f](?=\d{3}\b)")
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    """Extract numeric quantities while ignoring SKU identifiers and dates."""
    text = SKU_ID.sub(" ", text)
    text = ISO_DATE.sub(" ", text)
    text = WRITTEN_DATE.sub(" ", text)
    text = SPACED_THOUSANDS.sub("", text)
    return [float(value.replace(",", "")) for value in NUMBER.findall(text)]


def collect_numbers(data: Any) -> list[float]:
    """Collect numeric values recursively from dictionaries, lists and scalars."""
    if isinstance(data, bool):
        return []
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, str):
        return extract_numbers(data)
    if isinstance(data, dict):
        return collect_numbers(list(data.values()))
    if isinstance(data, (list, tuple)):
        return [number for item in data for number in collect_numbers(item)]
    return []


def check_numbers(
    text: str,
    allowed: list[float],
    rel_tol: float = 0.01,
    abs_tol: float = 0.05,
) -> tuple[bool, list[float]]:
    """Return whether every numeric quantity in text is grounded in allowed data."""
    unmatched = [
        number
        for number in extract_numbers(text)
        if not any(
            math.isclose(number, allowed_value, rel_tol=rel_tol, abs_tol=abs_tol)
            for allowed_value in allowed
        )
    ]
    return not unmatched, unmatched
