"""Text normalization and language detection for Nepali documents.

Cleans OCR and copy-paste artifacts before chunking and embedding.
"""

import unicodedata
import re


ZERO_WIDTH_CHARS = re.compile("[\u200b\u200c\u200d\ufeff]")

MULTI_SPACES = re.compile(r"[ \t]+")

MULTI_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize text for consistent chunking and embedding.

    Operations applied in order:
    1. Unicode NFC normalization
    2. Zero-width character removal
    3. Whitespace normalization (spaces/tabs collapsed, 3+ newlines to 2)
    4. Strip leading/trailing whitespace

    Args:
        text (str): Text to normalize.

    Returns:
        str: Normalized text.
    """
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_CHARS.sub("", text)
    text = MULTI_SPACES.sub(" ", text)
    text = MULTI_NEWLINES.sub("\n\n", text)
    text = text.strip()
    return text


def is_nepali_text(text: str) -> bool:
    """Return True if >60% of non-whitespace characters are Devanagari (U+0900–U+097F).
    
    Args:
        text (str): Text to check.
    
    Returns:
        bool: True if text is Nepali, False otherwise.
    """
    if not text:
        return False

    total = 0
    devanagari = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if "\u0900" <= ch <= "\u097f":
            devanagari += 1

    if total == 0:
        return False

    return (devanagari / total) > 0.60
