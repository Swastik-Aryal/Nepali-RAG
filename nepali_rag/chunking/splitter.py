"""Chunking utilities for splitting Nepali documents into token-limited segments.

Uses LangChain's :class:`RecursiveCharacterTextSplitter` with a Nepali-aware
separator hierarchy and token-based sizing:

* **Separators** — the splitter tries, in order, paragraph breaks (``\\n\\n``),
  line breaks (``\\n``), Nepali sentence delimiters (``।``, ``॥``), other
  terminators (``?``, ``!``), spaces, and finally raw characters. This keeps the
  splitter language-agnostic while still respecting Devanagari sentence
  boundaries (unlike English-trained sentence tokenizers).
* **Token sizing** — chunk size and overlap are measured with the embedding
  model's HuggingFace tokenizer, so chunks stay within the model's context
  window.
* **Overlap** — a percentage of ``chunk_size`` (see ``OVERLAP_RATIO``) is
  repeated between consecutive chunks to preserve context across boundaries.

Pages are concatenated before splitting so chunks can flow across page
boundaries; each chunk's ``page_number`` reflects the page(s) it spans (e.g.
``"1&2"`` for a chunk straddling pages 1 and 2).
"""

import logging

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from nepali_rag.config.settings import settings

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Fraction of ``chunk_size`` repeated between consecutive chunks as overlap.
OVERLAP_RATIO = 0.15

# Separator hierarchy tried in order. Nepali sentences end with the danda (।)
# and double danda (॥); "." is intentionally omitted because in Nepali it mostly
# appears inside numbers/abbreviations and splitting on it fragments badly.
NEPALI_SEPARATORS = ["\n\n", "\n", "।", "॥", "?", "!", " ", ""]

# Character used to join page texts before splitting.
_PAGE_JOINER = "\n\n"


def _make_tokenizer(embedding_model_name: str):
    """Load the embedding model's tokenizer used for token counting."""
    return AutoTokenizer.from_pretrained(embedding_model_name)


def _token_len(tokenizer):
    """Return a length function that counts tokens for the given tokenizer."""

    def _count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

    return _count


def _build_combined_text(pages):
    """Join page texts and record each page's character span in the result.

    Returns a tuple ``(combined_text, spans)`` where ``spans`` is a list of
    ``(start, end, page_number, source)`` describing where each non-empty page
    lives inside ``combined_text``.
    """
    parts = []
    spans = []
    offset = 0
    for page in pages:
        text = page.text or ""
        if not text.strip():
            continue
        start = offset
        end = offset + len(text)
        spans.append((start, end, page.page_number, page.source))
        parts.append(text)
        offset = end + len(_PAGE_JOINER)
    return _PAGE_JOINER.join(parts), spans


def _pages_for_span(chunk_start: int, chunk_end: int, spans):
    """Return ``(page_number, source)`` for the chunk covering ``[start, end)``.

    ``page_number`` is the single int page when the chunk lies within one page,
    or a ``"1&2"`` style string when it straddles several. Falls back to the
    first page if no overlap is detected (should not happen in practice).
    """
    pages = []
    source = None
    for start, end, page_number, page_source in spans:
        if start < chunk_end and end > chunk_start:
            if page_number not in pages:
                pages.append(page_number)
            if source is None:
                source = page_source
    if not pages:
        _s, _e, page_number, page_source = spans[0]
        return page_number, page_source
    if len(pages) == 1:
        return pages[0], source
    return "&".join(str(p) for p in pages), source


def chunk_pages(
    pages,
    chunk_size: int = None,
    chunk_overlap: int = None,
    embedding_model_name: str = None,
) -> list[dict]:
    """Split preprocessed pages into token-limited, overlapping chunks.

    Pages are concatenated and split with a Nepali-aware
    :class:`RecursiveCharacterTextSplitter`. Chunk size and overlap are measured
    in tokens using the embedding model's tokenizer.

    Args:
        pages: List of ``PageResult`` objects (from ingestion).
        chunk_size: Max tokens per chunk. Defaults to ``settings.chunk_size``.
        chunk_overlap: Tokens of overlap between consecutive chunks. Defaults to
            ``round(chunk_size * OVERLAP_RATIO)``.
        embedding_model_name: HuggingFace model name whose tokenizer is used for
            token counting. Defaults to ``settings.embedding_model``.

    Returns:
        List of chunk dicts, each with ``'text'`` (str) and ``'metadata'`` (dict)
        keys. Metadata includes ``source``, ``page_number``, ``chunk_index``,
        ``token_count``, and ``has_overlap``. ``page_number`` is the int page for
        single-page chunks, or a ``"1&2"`` style string for chunks spanning pages.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if embedding_model_name is None:
        embedding_model_name = settings.embedding_model
    if chunk_overlap is None:
        chunk_overlap = round(chunk_size * OVERLAP_RATIO)

    tokenizer = _make_tokenizer(embedding_model_name)

    if chunk_size > tokenizer.model_max_length:
        raise ValueError(
            f"chunk_size ({chunk_size}) exceeds the embedding model's max input length "
            f"({tokenizer.model_max_length}). Reduce chunk_size in your environment "
            f"variables or pass a smaller value."
        )
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )

    combined, spans = _build_combined_text(pages)
    if not combined.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=NEPALI_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_len(tokenizer),
        keep_separator="end",
        strip_whitespace=False,
    )

    raw_chunks = splitter.split_text(combined)

    all_chunks = []
    chunk_index = 0
    search_pos = 0
    prev_page_number = spans[0][2]
    prev_source = spans[0][3]
    for raw in raw_chunks:
        text = raw.strip()
        if not text:
            continue

        # Locate the chunk in the combined text (forward-only) to map it back to
        # source pages. Chunks are contiguous substrings, so a forward search
        # correctly resolves even overlapping regions.
        pos = combined.find(raw, search_pos)
        if pos == -1:
            page_number, source = prev_page_number, prev_source
        else:
            page_number, source = _pages_for_span(pos, pos + len(raw), spans)
            search_pos = pos + 1
            prev_page_number, prev_source = page_number, source

        all_chunks.append({
            "text": text,
            "metadata": {
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                "has_overlap": chunk_overlap > 0 and chunk_index > 0,
            },
        })
        chunk_index += 1

    return all_chunks
