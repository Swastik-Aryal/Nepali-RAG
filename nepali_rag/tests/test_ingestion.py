import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nepali_rag.ingestion.loader import load_document
from nepali_rag.ingestion.ocr_parser import ocr_all_pages
from nepali_rag.ingestion.models import PageResult


def test_pdf_ocr_extraction():
    """Test OCR text extraction from all pages of a PDF."""
    test_pdf = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data","sample_docs","test.pdf"
    )
    if not os.path.exists(test_pdf):
        print(f"[SKIP] test.pdf not found at {test_pdf}")
        return

    results = ocr_all_pages(test_pdf)

    print(f"\nOCR extraction results:")
    print(f"  Pages extracted: {len(results)}")

    for r in results:
        assert isinstance(r, PageResult)
        assert r.extraction_method == "ocr"
        assert r.page_number >= 1
        text_preview = r.text[:100] if r.text else "(empty)"
        print(f"  Page {r.page_number}: {len(r.text)} chars | {text_preview}...")

    print("[PASS] PDF OCR extraction")


def test_load_document_pdf():
    """Test the unified loader with PDF (OCR for all pages)."""
    test_pdf = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data","sample_docs","test1.pdf"
    )
    if not os.path.exists(test_pdf):
        print(f"[SKIP] test.pdf not found at {test_pdf}")
        return

    results = load_document(test_pdf)

    print(f"\nUnified loader results:")
    print(f"  Total pages: {len(results)}")

    for r in results:
        assert isinstance(r, PageResult)
        assert r.page_number >= 1
        assert r.source
        assert r.extraction_method == "ocr"
        has_text = bool(r.text)
        print(
            f"  Page {r.page_number}: method={r.extraction_method} | "
            f"has_text={has_text} | {len(r.text)} chars"
        )

    pages_with_text = sum(1 for r in results if r.text)
    print(f"  Pages with text: {pages_with_text}/{len(results)}")
    print(f"  All pages processed via OCR: {len(results)}")
    print("[PASS] Unified loader PDF")


def test_load_document_txt():
    """Test the unified loader with a plain text file."""
    test_dir = os.path.dirname(os.path.dirname(__file__))
    test_txt = os.path.join(test_dir, "data", "sample_docs", "test_nepali.txt")

    os.makedirs(os.path.dirname(test_txt), exist_ok=True)

    sample_text = "नेपाल एक सुन्दर देश हो। यो हिमालयको देश हो।"
    with open(test_txt, "w", encoding="utf-8") as f:
        f.write(sample_text)

    results = load_document(test_txt)

    assert len(results) == 1
    assert results[0].page_number == 1
    assert results[0].extraction_method == "text_file"
    assert results[0].text == sample_text
    print(f"\nTXT loader result:")
    print(f"  Text: {results[0].text}")
    print("[PASS] TXT loader")


def test_page_result_metadata():
    """Test that PageResult carries all required metadata."""
    pr = PageResult(
        page_number=1,
        text="test text",
        source="/path/to/file.pdf",
        extraction_method="ocr"
    )
    assert pr.page_number == 1
    assert pr.text == "test text"
    assert pr.source == "/path/to/file.pdf"
    assert pr.extraction_method == "ocr"
    print("\n[PASS] PageResult metadata")


def test_unsupported_format():
    """Test that truly unsupported formats raise an error."""
    try:
        load_document("test.xyz")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n[PASS] Unsupported format error: {e}")


def test_load_docx():
    """Test DOCX loader."""
    from docx import Document

    test_dir = os.path.dirname(os.path.dirname(__file__))
    test_docx = os.path.join(test_dir, "data", "sample_docs", "test.docx")
    
    results = load_document(test_docx)
    assert len(results) == 1
    assert results[0].extraction_method == "docx"
    assert "नेपाल" in results[0].text
    print(f"\n  DOCX text: {results[0].text[:80]}")
    print("[PASS] DOCX loader")


def test_load_xlsx():
    """Test XLSX loader."""
    from openpyxl import Workbook

    test_dir = os.path.dirname(os.path.dirname(__file__))
    test_xlsx = os.path.join(test_dir, "data", "sample_docs", "test.xlsx")
    os.makedirs(os.path.dirname(test_xlsx), exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Name", "City"])
    ws.append(["राम", "काठमाडौं"])
    wb.save(test_xlsx)

    results = load_document(test_xlsx)
    assert len(results) == 1
    assert results[0].extraction_method == "xlsx"
    assert "राम" in results[0].text
    print(f"\n  XLSX text: {results[0].text[:80]}")
    print("[PASS] XLSX loader")


def test_load_csv():
    """Test CSV loader."""
    test_dir = os.path.dirname(os.path.dirname(__file__))
    test_csv = os.path.join(test_dir, "data", "sample_docs", "test.csv")
    os.makedirs(os.path.dirname(test_csv), exist_ok=True)

    with open(test_csv, "w", encoding="utf-8", newline="") as f:
        f.write("Name,City\n")
        f.write("राम,काठमाडौं\n")

    results = load_document(test_csv)
    assert len(results) == 1
    assert results[0].extraction_method == "csv"
    assert "राम" in results[0].text
    print(f"\n  CSV text: {results[0].text[:80]}")
    print("[PASS] CSV loader")


def test_load_html():
    """Test HTML loader."""
    test_dir = os.path.dirname(os.path.dirname(__file__))
    test_html = os.path.join(test_dir, "data", "sample_docs", "test.html")
    os.makedirs(os.path.dirname(test_html), exist_ok=True)

    html_content = """<html><body>
    <h1>नेपाल</h1>
    <p>नेपाल एक सुन्दर देश हो।</p>
    <script>var x = 1;</script>
    </body></html>"""
    with open(test_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    results = load_document(test_html)
    assert len(results) == 1
    assert results[0].extraction_method == "html"
    assert "नेपाल" in results[0].text
    assert "var x" not in results[0].text
    print(f"\n  HTML text: {results[0].text[:80]}")
    print("[PASS] HTML loader")


def test_load_md():
    """Test Markdown loader (treated as plain text)."""
    test_dir = os.path.dirname(os.path.dirname(__file__))
    test_md = os.path.join(test_dir, "data", "sample_docs", "test.md")
    os.makedirs(os.path.dirname(test_md), exist_ok=True)

    md_content = "# नेपाल\n\nनेपाल एक सुन्दर देश हो।"
    with open(test_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    results = load_document(test_md)
    assert len(results) == 1
    assert results[0].extraction_method == "text_file"
    assert "नेपाल" in results[0].text
    print(f"\n  MD text: {results[0].text[:80]}")
    print("[PASS] Markdown loader")


if __name__ == "__main__":
    print("COMPONENT 1: Document Ingestion Tests")

    test_page_result_metadata()
    test_unsupported_format()
    test_load_document_txt()
    test_load_docx()
    test_load_xlsx()
    test_load_csv()
    test_load_html()
    test_load_md()
    test_pdf_ocr_extraction()
    test_load_document_pdf()

    print("All ingestion tests completed.")

