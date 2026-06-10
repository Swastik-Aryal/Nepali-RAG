from nepali_rag.ingestion.models import PageResult
from nepali_rag.chunking.splitter import chunk_pages, _count_tokens, _split_sentences


def _make_page(text, source="test.pdf", page_number=1):
    return PageResult(page_number=page_number, text=text, source=source, extraction_method="text_file")


def test_short_paragraph_stays_intact():
    text = "नेपाल एक सुन्दर देश हो। यहाँ हिमालय छ।"
    pages = [_make_page(text)]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    assert len(chunks) == 1
    assert chunks[0]["text"].strip() == text.strip()
    assert chunks[0]["metadata"]["has_overlap"] is False
    assert chunks[0]["metadata"]["token_count"] <= 512


def test_large_paragraph_splits_at_sentence_boundary():
    sentence = "यो एउटा लामो वाक्य हो जसलाई परीक्षणको लागि प्रयोग गरिएको छ।"
    s_tokens = _count_tokens(sentence)
    repeat_count = (512 // s_tokens) + 5
    text = " ".join([sentence] * repeat_count)
    pages = [_make_page(text)]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "।" in chunk["text"] or chunk["metadata"]["token_count"] <= 512


def test_no_chunk_exceeds_512_tokens():
    sentence = "यो एउटा वाक्य हो।"
    text = " ".join([sentence] * 200)
    pages = [_make_page(text)]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    for chunk in chunks:
        actual_tokens = _count_tokens(chunk["text"])
        assert actual_tokens <= 512, f"Chunk has {actual_tokens} tokens, expected <= 512"


def test_overlap_applied_on_split_chunks():
    sentence = "यो वाक्य नम्बर {}।"
    sentences = [sentence.format(i) for i in range(1, 101)]
    text = " ".join(sentences)
    pages = [_make_page(text)]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    if len(chunks) > 1:
        assert chunks[0]["metadata"]["has_overlap"] is False
        for chunk in chunks[1:]:
            assert chunk["metadata"]["has_overlap"] is True


def test_force_split_long_sentence():
    word = "अत्यन्तलामोशब्द"
    long_sentence = " ".join([word] * 300)
    pages = [_make_page(long_sentence)]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert _count_tokens(chunk["text"]) <= 512


def test_small_blocks_merged():
    block1 = "छोटो।"
    block2 = "अर्को छोटो।"
    text = block1 + "\n\n" + block2
    pages = [_make_page(text)]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    assert len(chunks) == 1


def test_metadata_fields():
    text = "नेपाली पाठ।"
    pages = [_make_page(text, source="doc.pdf", page_number=3)]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    assert len(chunks) == 1
    meta = chunks[0]["metadata"]
    assert meta["source"] == "doc.pdf"
    assert meta["page_number"] == 3
    assert meta["chunk_index"] == 0
    assert "token_count" in meta
    assert "has_overlap" in meta


def test_sentence_splitting_keeps_delimiter():
    text = "पहिलो वाक्य। दोस्रो वाक्य। तेस्रो वाक्य।"
    sentences = _split_sentences(text)
    for s in sentences:
        assert s.endswith("।")


def test_multiple_pages():
    pages = [
        _make_page("पहिलो पृष्ठको पाठ।", page_number=1),
        _make_page("दोस्रो पृष्ठको पाठ।", page_number=2),
    ]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["page_number"] == 1
    assert chunks[1]["metadata"]["page_number"] == 2
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["metadata"]["chunk_index"] == 1


def test_empty_page_skipped():
    pages = [_make_page(""), _make_page("  "), _make_page("कुनै पाठ।")]
    chunks = chunk_pages(pages, chunk_size=512, overlap_sentences=2)
    assert len(chunks) == 1


if __name__ == "__main__":
    tests = [
        test_short_paragraph_stays_intact,
        test_large_paragraph_splits_at_sentence_boundary,
        test_no_chunk_exceeds_512_tokens,
        test_overlap_applied_on_split_chunks,
        test_force_split_long_sentence,
        test_small_blocks_merged,
        test_metadata_fields,
        test_sentence_splitting_keeps_delimiter,
        test_multiple_pages,
        test_empty_page_skipped,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__} -> {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
