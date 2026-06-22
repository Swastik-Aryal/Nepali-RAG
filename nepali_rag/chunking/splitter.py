"""Chunking utilities for splitting Nepali documents into token-limited segments.

Provides a sentence-aware balancing pipeline:

1. **Structural split** — recursively splits each page on paragraph/line
   separators (``\\n\\n`` then ``\\n``) to seed initial chunks, keeping whole
   sentences intact.
2. **Balancing pass** — walks chunks front-to-back and enforces two rules
   until both hold for every chunk:

   * *Too big* (``> chunk_size``): trailing whole sentences are pushed into the
     next chunk; a lone oversized sentence is force-split at the token boundary
     with the full sentence repeated at the start of the next chunk.
   * *Too small* (``< chunk_size / 2``): the next chunk is merged into the
     current one (across page boundaries when needed).

   Sentences are only ever split mid-way in a rare edge case (a single large
   sentence straddling a boundary), in which case the full sentence is repeated
   at the start of the following chunk.

Token counting uses the HuggingFace tokenizer for the configured embedding model
to ensure chunks stay within the model's context window.
"""

import re
from transformers import AutoTokenizer
from nepali_rag.config.settings import settings
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # for the token length error during checking phase

_tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

SENTENCE_DELIMITERS = re.compile(r"([।॥!?])")
PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n")


def _ends_with_delimiter(text: str) -> bool:
    """Return *True* if *text* ends with a sentence-ending delimiter."""
    text = text.rstrip()
    if not text:
        return True  # empty text is not a fragment
    return bool(SENTENCE_DELIMITERS.match(text[-1]))


def _count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using the embedding model's tokenizer."""
    return len(_tokenizer.encode(text, add_special_tokens=False, truncation=False))


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


def _force_split_tokens(text: str, max_tokens: int) -> list[str]:
    """Force-split a single oversized sentence at the token boundary (last resort).

    Returns a list of pieces where each fits within *max_tokens*. This is only
    used when a single sentence exceeds the chunk size and cannot be split any
    other way.
    """
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    pieces = []
    for start in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[start : start + max_tokens]
        piece = _tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if piece:
            pieces.append(piece)
    return pieces or [text.strip()]


def _split_sentence_half(text: str, budget_tokens: int) -> str:
    """Return the leading portion of *text* that fits within *budget_tokens*.

    Used only for the rare mid-sentence split, where a partial copy of a large
    sentence is used to top up an otherwise undersized chunk. The full sentence
    is kept (and repeated) in the following chunk.
    """
    if budget_tokens <= 0:
        return ""
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    prefix = _tokenizer.decode(token_ids[:budget_tokens], skip_special_tokens=True).strip()
    return prefix


class _Sent:
    """A single sentence carrying the page/source it originated from."""

    __slots__ = ("text", "page", "source")

    def __init__(self, text: str, page, source):
        self.text = text
        self.page = page
        self.source = source


class _Chunk:
    """A mutable group of sentences that will become one output chunk."""

    __slots__ = ("sents", "has_overlap")

    def __init__(self, sents: list[_Sent], has_overlap: bool = False):
        self.sents = sents
        self.has_overlap = has_overlap


def _chunk_text(chunk: _Chunk) -> str:
    """Join a chunk's sentences back into a single text block."""
    return " ".join(s.text for s in chunk.sents).strip()


def _chunk_tokens(chunk: _Chunk) -> int:
    """Token count of a chunk's rendered text."""
    return _count_tokens(_chunk_text(chunk))


def _structural_blocks(text: str, chunk_size: int) -> list[str]:
    """Recursively split *text* on paragraph then line boundaries.

    Paragraphs that already fit within *chunk_size* are kept whole; oversized
    paragraphs are further broken on single newlines. Sentence integrity is
    never violated here — anything still oversized is handled later by the
    balancing pass.
    """
    blocks = []
    for paragraph in PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _count_tokens(paragraph) <= chunk_size:
            blocks.append(paragraph)
        else:
            for line in paragraph.split("\n"):
                line = line.strip()
                if line:
                    blocks.append(line)
    return blocks


def _initial_chunks(pages, chunk_size: int) -> list[_Chunk]:
    """Build seed chunks from the structural split, one per paragraph/line block.

    Sentence fragments at the end of one structural block are joined with the
    first sentence of the following block, so the balancing pass always works
    with complete sentences.  This prevents mid-sentence splits at chunk
    boundaries: every sentence moved between chunks during balancing is a
    whole, delimiter-terminated sentence.
    """
    chunks = []
    # A pending sentence fragment whose text did not end with a delimiter.
    # Stored as ``(text, page_number, source)``.
    pending_fragment = None

    for page in pages:
        text = page.text
        if not text or not text.strip():
            continue
        for block in _structural_blocks(text, chunk_size):
            sents = _split_sentences(block)
            if not sents:
                continue

            # Prepend any pending fragment to the first sentence of this block.
            if pending_fragment is not None:
                frag_text, _frag_page, _frag_source = pending_fragment
                sents[0] = frag_text + " " + sents[0]
                pending_fragment = None

            # If the last sentence does not end with a delimiter it is a
            # fragment — stash it and do not include it in the chunk yet.
            if not _ends_with_delimiter(sents[-1]):
                pending_fragment = (sents.pop(), page.page_number, page.source)
                if not sents:
                    continue  # entire block was one fragment

            chunk_sents = [
                _Sent(s, page.page_number, page.source) for s in sents
            ]
            chunks.append(_Chunk(chunk_sents))

    # Flush any remaining fragment as the last sentence of the last chunk.
    if pending_fragment is not None:
        frag_text, frag_page, frag_source = pending_fragment
        frag_sent = _Sent(frag_text, frag_page, frag_source)
        if chunks:
            chunks[-1].sents.append(frag_sent)
        else:
            chunks.append(_Chunk([frag_sent]))

    return chunks


def _balance_chunks(chunks: list[_Chunk], chunk_size: int, min_tokens: int) -> list[_Chunk]:
    """Apply the two balancing rules front-to-back until both hold everywhere.

    Rule 1 (too big): push trailing whole sentences into the next chunk until
    the current chunk fits; if a single sentence still overflows, split it
    in half — keep the first half here, and repeat the full sentence at the
    start of the next chunk (overlap). Only as a last resort, force-split a
    truly massive sentence on raw token boundaries.

    Rule 2 (too small): merge the following chunk into the current one and
    re-check, which naturally bridges page boundaries.
    """
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        cur_tokens = _chunk_tokens(cur)

        # Rule 1: chunk exceeds the size limit
        if cur_tokens > chunk_size:

            # 1a. Multi-sentence: push trailing sentences forward.
            if len(cur.sents) > 1:
                # Ensure a next chunk exists to receive pushed sentences.
                if i + 1 >= len(chunks):
                    chunks.append(_Chunk([]))
                nxt = chunks[i + 1]

                while _chunk_tokens(cur) > chunk_size and len(cur.sents) > 1:
                    nxt.sents.insert(0, cur.sents.pop())

                cur_tokens = _chunk_tokens(cur)

                # After pushing, current chunk may be too small.
                if cur_tokens < min_tokens and nxt.sents:
                    # Edge case: we pushed sentences away and now we're too
                    # small. Top up with a partial (half) copy of the next
                    # chunk's leading sentence. The full sentence stays in
                    # the next chunk (→ overlap).
                    budget = chunk_size - cur_tokens
                    prefix = _split_sentence_half(nxt.sents[0].text, budget)
                    if prefix:
                        cur.sents.append(
                            _Sent(prefix, nxt.sents[0].page, nxt.sents[0].source)
                        )
                        nxt.has_overlap = True
                    # Don't re-check this chunk; the prefix fill is a
                    # controlled approximation. Move on.

                i += 1
                continue

            # 1b. Single sentence that overflows.
            lead = cur.sents[0]
            lead_tokens = _count_tokens(lead.text)

            if lead_tokens <= chunk_size * 2:
                # Sentence is at most 2× chunk_size: split it in half.
                # Keep first half here, put the FULL sentence as overlap in
                # the next chunk so context is preserved.
                prefix = _split_sentence_half(lead.text, chunk_size)
                if prefix:
                    cur.sents = [_Sent(prefix, lead.page, lead.source)]

                    # Ensure a next chunk exists.
                    if i + 1 >= len(chunks):
                        chunks.append(_Chunk([]))
                    nxt = chunks[i + 1]
                    # Repeat the full sentence at the start of the next chunk.
                    nxt.sents.insert(0, _Sent(lead.text, lead.page, lead.source))
                    nxt.has_overlap = True
                    i += 1
                    continue

            # Truly massive single sentence (> 2× chunk_size): hard split on
            # token boundaries. Each piece becomes its own chunk.
            pieces = _force_split_tokens(lead.text, chunk_size)
            cur.sents = [_Sent(pieces[0], lead.page, lead.source)]
            insert_at = i + 1
            for piece in pieces[1:]:
                new_chunk = _Chunk(
                    [_Sent(piece, lead.page, lead.source)], has_overlap=True
                )
                chunks.insert(insert_at, new_chunk)
                insert_at += 1
            i += 1
            continue

        # Rule 2: chunk is below half-size — pull next chunk in
        if cur_tokens < min_tokens and i + 1 < len(chunks):
            cur.sents.extend(chunks[i + 1].sents)
            del chunks[i + 1]
            # Re-check this index (the merged result may now be too big).
            continue

        i += 1

    return chunks


def _page_number(chunk: _Chunk):
    """Derive the chunk's page number, joining with ``&`` when it spans pages."""
    pages = []
    for s in chunk.sents:
        if s.page not in pages:
            pages.append(s.page)
    if len(pages) == 1:
        return pages[0]
    return "&".join(str(p) for p in pages)


def chunk_pages(
    pages,
    chunk_size: int = None
) -> list[dict]:
    """Split preprocessed pages into token-limited chunks.

    Two-stage process:
    Stage 1: Recursive structural split on paragraph/line boundaries (\\n\\n, \\n)
             seeds initial chunks while keeping whole sentences intact.
    Stage 2: A balancing pass walks the chunks and enforces, until both hold:
             oversized chunks (> chunk_size) shed trailing sentences forward,
             and undersized chunks (< chunk_size / 2) absorb the next chunk —
             merging across page boundaries when required. Sentences are split
             mid-way only in rare edge cases, and the full sentence is then
             repeated at the start of the following chunk.

    Args:
        pages: List of PageResult objects (from ingestion).
        chunk_size: Max tokens per chunk. Defaults to ``settings.chunk_size`` (256).

    Returns:
        List of chunk dicts, each with ``'text'`` (str) and ``'metadata'`` (dict)
        keys. Metadata includes ``source``, ``page_number``, ``chunk_index``,
        ``token_count``, and ``has_overlap``. ``page_number`` is the int page for
        single-page chunks, or a ``"1&2"`` style string for chunks spanning pages.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size

    if chunk_size > _tokenizer.model_max_length:
        raise ValueError(
            f"chunk_size ({chunk_size}) exceeds the embedding model's max input length "
            f"({_tokenizer.model_max_length}). Reduce chunk_size in your environment variables or pass a smaller value."
        )

    min_tokens = chunk_size // 2

    chunks = _initial_chunks(pages, chunk_size)
    chunks = _balance_chunks(chunks, chunk_size, min_tokens)

    all_chunks = []
    chunk_index = 0
    for chunk in chunks:
        text = _chunk_text(chunk)
        if not text:
            continue
        all_chunks.append({
            "text": text,
            "metadata": {
                "source": chunk.sents[0].source,
                "page_number": _page_number(chunk),
                "chunk_index": chunk_index,
                "token_count": _count_tokens(text),
                "has_overlap": chunk.has_overlap,
            },
        })
        chunk_index += 1

    return all_chunks
