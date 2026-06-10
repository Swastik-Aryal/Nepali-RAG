import unicodedata
import re


ZERO_WIDTH_CHARS = re.compile("[\u200b\u200c\u200d\ufeff]")

MULTI_SPACES = re.compile(r"[ \t]+")

MULTI_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize Nepali text for consistent representation.

    Operations applied in order:
    1. Unicode NFC normalization
    2. Zero-width character removal
    3. Whitespace normalization (spaces/tabs collapsed, 3+ newlines to 2)
    4. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_CHARS.sub("", text)
    text = MULTI_SPACES.sub(" ", text)
    text = MULTI_NEWLINES.sub("\n\n", text)
    text = text.strip()
    return text


def is_nepali_text(text: str) -> bool:
    """Check whether a string is primarily Devanagari.

    Returns True if more than 30% of characters fall in the
    Devanagari Unicode range (U+0900 to U+097F).
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

    return (devanagari / total) > 0.30
