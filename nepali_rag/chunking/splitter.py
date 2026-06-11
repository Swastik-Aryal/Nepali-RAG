"""Chunking utilities for splitting Nepali documents into token-limited segments.

Provides a two-level splitting pipeline:

1. **Structural split** — uses paragraph/line separators to keep logical sections
   intact, then trims each block to end on a sentence boundary.
2. **Sentence-level split** — breaks oversized blocks at Nepali sentence
   delimiters (purna viram ``।``, double danda ``॥``, ``!``, ``?``, ``.``) with
   configurable sentence overlap between consecutive chunks.

Token counting uses the HuggingFace tokenizer for the configured embedding model
to ensure chunks stay within the model's context window.
"""

import re
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nepali_rag.config.settings import settings

_tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

SENTENCE_DELIMITERS = re.compile(r"([।॥!?.])")


def _count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using the embedding model's tokenizer."""
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence delimiters, keeping delimiters attached to the preceding sentence."""
    parts = SENTENCE_DELIMITERS.split(text)
    sentences = []
    i = 0
    while i < len(parts):
        segment = parts[i]
        if i + 1 < len(parts) and SENTENCE_DELIMITERS.match(parts[i + 1]):
            segment += parts[i + 1]
            i += 2
        else:
            i += 1
        segment = segment.strip()
        if segment:
            sentences.append(segment)
    return sentences


def _force_split(text: str, max_tokens: int) -> list[str]:
    """Force-split a single chunk that exceeds max_tokens at the token boundary."""
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[start : start + max_tokens]
        chunk_text = _tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def _sentence_chunk(text: str, max_tokens: int, overlap_sentences: int) -> list[dict]:
    """Split text into chunks at sentence boundaries (Level 2).

    Greedily accumulates sentences until *max_tokens* is reached, then starts
    a new chunk with *overlap_sentences* carried over for continuity.

    Returns:
        List of dicts with keys ``text``, ``token_count``, and ``has_overlap``.
    """
    sentences = _split_sentences(text)
    if not sentences:
        if not text.strip():
            return []
        tokens = _count_tokens(text)
        if tokens <= max_tokens:
            return [{"text": text.strip(), "token_count": tokens, "has_overlap": False}]
        chunks = []
        for piece in _force_split(text, max_tokens):
            chunks.append({"text": piece, "token_count": _count_tokens(piece), "has_overlap": False})
        return chunks

    all_chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _count_tokens(sentence)

        if s_tokens > max_tokens:
            if current_sentences:
                combined = " ".join(current_sentences)
                all_chunks.append({
                    "text": combined,
                    "token_count": _count_tokens(combined),
                    "has_overlap": bool(all_chunks),
                })
                current_sentences = []
                current_tokens = 0

            for piece in _force_split(sentence, max_tokens):
                all_chunks.append({
                    "text": piece,
                    "token_count": _count_tokens(piece),
                    "has_overlap": False,
                })
            continue

        combined_text = " ".join(current_sentences + [sentence])
        combined_tokens = _count_tokens(combined_text)

        if combined_tokens <= max_tokens:
            current_sentences.append(sentence)
            current_tokens = combined_tokens
        else:
            if current_sentences:
                combined = " ".join(current_sentences)
                all_chunks.append({
                    "text": combined,
                    "token_count": _count_tokens(combined),
                    "has_overlap": False,
                })

                overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
                current_sentences = overlap + [sentence]
                combined = " ".join(current_sentences)
                current_tokens = _count_tokens(combined)

                if current_tokens > max_tokens:
                    current_sentences = [sentence]
                    current_tokens = s_tokens
            else:
                current_sentences = [sentence]
                current_tokens = s_tokens

    if current_sentences:
        combined = " ".join(current_sentences)
        has_overlap = len(all_chunks) > 0
        all_chunks.append({
            "text": combined,
            "token_count": _count_tokens(combined),
            "has_overlap": has_overlap,
        })

    if len(all_chunks) > 1:
        all_chunks[0]["has_overlap"] = False
        for i in range(1, len(all_chunks)):
            all_chunks[i]["has_overlap"] = True

    return all_chunks


def _merge_small_blocks(blocks: list[str], min_tokens: int, max_tokens: int) -> list[str]:
    """Merge undersized blocks with their next neighbor to avoid tiny fragments.

    Blocks below *min_tokens* are concatenated with the following block,
    as long as the combined result stays within *max_tokens*.
    """
    if not blocks:
        return blocks

    merged = []
    buffer = blocks[0]

    for i in range(1, len(blocks)):
        buf_tokens = _count_tokens(buffer)
        if buf_tokens < min_tokens:
            candidate = buffer + "\n" + blocks[i]
            if _count_tokens(candidate) <= max_tokens:
                buffer = candidate
                continue
        merged.append(buffer)
        buffer = blocks[i]

    merged.append(buffer)
    return merged


def _ends_with_sentence(text: str) -> bool:
    """Check if text ends with a sentence delimiter (।, ॥, !, ?, .)."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return bool(SENTENCE_DELIMITERS.match(stripped[-1]))


def _trim_to_sentence_boundary(text: str) -> tuple[str, str]:
    """Split text into (complete_sentences, leftover_fragment).

    Finds the last sentence delimiter in the text and splits there.
    Returns (text_up_to_and_including_last_delimiter, remaining_text).
    If no delimiter is found, returns ("", text) — meaning the whole text
    is a fragment.
    """
    # Find the last occurrence of any sentence delimiter
    last_pos = -1
    for m in SENTENCE_DELIMITERS.finditer(text):
        last_pos = m.end()

    if last_pos == -1:
        # No sentence delimiter found at all — whole text is a fragment
        return "", text

    complete = text[:last_pos].rstrip()
    leftover = text[last_pos:].strip()
    return complete, leftover


def _ensure_sentence_boundaries(blocks: list[str]) -> list[str]:
    """Post-process blocks so every block ends on a sentence boundary.

    If a block doesn't end with a sentence delimiter, the trailing
    incomplete sentence is trimmed off and prepended to the next block.
    """
    if not blocks:
        return blocks

    result = []
    carry = ""  # leftover fragment from the previous block

    for i, block in enumerate(blocks):
        # Prepend any leftover from the previous block
        if carry:
            block = carry + " " + block
            carry = ""

        block = block.strip()
        if not block:
            continue

        if _ends_with_sentence(block):
            result.append(block)
        else:
            complete, leftover = _trim_to_sentence_boundary(block)
            if complete:
                result.append(complete)
            # Carry the leftover to the next block
            carry = leftover

    # If there's leftover after the last block, emit it as a separate block
    # rather than appending to the previous chunk (which would break its
    # sentence boundary). This leftover is typically footer/glossary text
    # from OCR that has no sentence delimiter.
    if carry:
        carry = carry.strip()
        if carry:
            result.append(carry)

    return [b for b in result if b.strip()]


def chunk_pages(
    pages,
    chunk_size: int = None,
    overlap_sentences: int = None,
) -> list[dict]:
    """Split preprocessed pages into token-limited chunks.

    Two-level process:
    Level 1: RecursiveCharacterTextSplitter uses structural separators (\n\n, \n)
             to keep paragraphs/sections intact, then a sentence-boundary pass
             ensures every block ends on a complete sentence.
    Level 2: Oversized blocks are split at sentence boundaries.

    Args:
        pages: List of PageResult objects (from ingestion).
        chunk_size: Max tokens per chunk. Defaults to ``settings.chunk_size`` (256).
        overlap_sentences: Number of sentences to overlap between consecutive
            chunks. Defaults to ``settings.chunk_overlap_sentences`` (2).

    Returns:
        List of chunk dicts, each with ``'text'`` (str) and ``'metadata'`` (dict)
        keys. Metadata includes ``source``, ``page_number``, ``chunk_index``,
        ``token_count``, and ``has_overlap``.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap_sentences is None:
        overlap_sentences = settings.chunk_overlap_sentences

    min_tokens = chunk_size // 2

    # Level 1: structural split on paragraph/line boundaries only
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=_count_tokens,
        keep_separator=False,
    )

    all_chunks = []
    chunk_index = 0

    for page in pages:
        text = page.text
        if not text or not text.strip():
            continue

        try:
            blocks = splitter.split_text(text)
            blocks = _merge_small_blocks(blocks, min_tokens, chunk_size)
        except Exception:
            blocks = [text]

        # Ensure every block ends on a sentence boundary
        blocks = _ensure_sentence_boundaries(blocks)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            block_tokens = _count_tokens(block)

            if block_tokens <= chunk_size:
                all_chunks.append({
                    "text": block,
                    "metadata": {
                        "source": page.source,
                        "page_number": page.page_number,
                        "chunk_index": chunk_index,
                        "token_count": block_tokens,
                        "has_overlap": False,
                    },
                })
                chunk_index += 1
            else:
                sub_chunks = _sentence_chunk(block, chunk_size, overlap_sentences)
                for sc in sub_chunks:
                    all_chunks.append({
                        "text": sc["text"],
                        "metadata": {
                            "source": page.source,
                            "page_number": page.page_number,
                            "chunk_index": chunk_index,
                            "token_count": sc["token_count"],
                            "has_overlap": sc["has_overlap"],
                        },
                    })
                    chunk_index += 1

    return all_chunks
