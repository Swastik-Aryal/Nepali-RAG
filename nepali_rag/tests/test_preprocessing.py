import unicodedata
from nepali_rag.preprocessing.normalizer import normalize_text, is_nepali_text


def test_nfc_normalization():
    decomposed = "e\u0301"
    composed = "\u00e9"
    assert decomposed != composed
    assert normalize_text(decomposed) == composed

    nepali_with_decomposed = "नमस्ते e\u0301 world"
    result = normalize_text(nepali_with_decomposed)
    assert "\u00e9" in result


def test_nfc_idempotent_on_nepali():
    text = "नेपाली भाषा"
    result = normalize_text(text)
    assert result == text


def test_zero_width_removal():
    text = "नमस्ते\u200b \u200cसंसार\u200d\ufeff"
    result = normalize_text(text)
    assert "\u200b" not in result
    assert "\u200c" not in result
    assert "\u200d" not in result
    assert "\ufeff" not in result
    assert result == "नमस्ते संसार"


def test_whitespace_collapse():
    text = "hello   world\t\there"
    result = normalize_text(text)
    assert result == "hello world here"


def test_newline_collapse():
    text = "paragraph one\n\n\n\n\nparagraph two"
    result = normalize_text(text)
    assert result == "paragraph one\n\nparagraph two"


def test_double_newline_preserved():
    text = "paragraph one\n\nparagraph two"
    result = normalize_text(text)
    assert result == "paragraph one\n\nparagraph two"


def test_strip():
    text = "   hello   "
    result = normalize_text(text)
    assert result == "hello"


def test_combined():
    text = "  \u200b नमस्ते \u200c  \n\n\n\n  world  \ufeff  "
    result = normalize_text(text)
    assert "\u200b" not in result
    assert "\u200c" not in result
    assert "\ufeff" not in result
    assert result == "नमस्ते \n\n world"


def test_is_nepali_pure_nepali():
    assert is_nepali_text("नेपाली भाषा राम्रो छ") is True


def test_is_nepali_pure_english():
    assert is_nepali_text("This is English text") is False


def test_is_nepali_mixed_majority_nepali():
    assert is_nepali_text("यो एउटा test हो") is True


def test_is_nepali_mixed_majority_english():
    assert is_nepali_text("This is a test with नेपाली") is False


def test_is_nepali_empty():
    assert is_nepali_text("") is False


def test_is_nepali_whitespace_only():
    assert is_nepali_text("   ") is False


if __name__ == "__main__":
    tests = [
        test_nfc_normalization,
        test_nfc_idempotent_on_nepali,
        test_zero_width_removal,
        test_whitespace_collapse,
        test_newline_collapse,
        test_double_newline_preserved,
        test_strip,
        test_combined,
        test_is_nepali_pure_nepali,
        test_is_nepali_pure_english,
        test_is_nepali_mixed_majority_nepali,
        test_is_nepali_mixed_majority_english,
        test_is_nepali_empty,
        test_is_nepali_whitespace_only,
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
