"""Document loading entry point for the ingestion pipeline.

Routes files by extension to the appropriate parser. PDFs are always processed
via Tesseract OCR at 300 DPI with Nepali language support; other formats use
format-specific extractors that return one or more ``PageResult`` objects.
"""

import json
import sys
import os
from dataclasses import asdict
from pathlib import Path

from nepali_rag.ingestion.models import PageResult
from nepali_rag.ingestion.ocr_parser import ocr_all_pages
from nepali_rag.ingestion.doc_parser import (
    load_docx,
    load_pptx,
    load_xlsx,
    load_csv_file,
    load_rtf,
    load_html,
    load_epub,
    load_odt,
)

SUPPORTED_FORMATS = {
    ".pdf": "PDF documents",
    ".txt": "Plain text files",
    ".md": "Markdown files",
    ".docx": "Microsoft Word documents",
    ".pptx": "Microsoft PowerPoint presentations",
    ".xlsx": "Microsoft Excel spreadsheets",
    ".csv": "CSV files",
    ".rtf": "Rich Text Format files",
    ".html": "HTML files",
    ".htm": "HTML files",
    ".epub": "EPUB ebooks",
    ".odt": "OpenDocument Text files",
}

FORMAT_HANDLERS = {
    ".pdf": "_load_pdf",
    ".txt": "_load_txt",
    ".md": "_load_txt",
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".xlsx": load_xlsx,
    ".csv": load_csv_file,
    ".rtf": load_rtf,
    ".html": load_html,
    ".htm": load_html,
    ".epub": load_epub,
    ".odt": load_odt,
}


def load_document(file_path: str) -> list[PageResult]:
    """Load a document and extract text from all pages/slides/sheets.

    Args:
        file_path: Path to a supported file (``.pdf``, ``.txt``, ``.md``,
            ``.docx``, ``.pptx``, ``.xlsx``, ``.csv``, ``.rtf``, ``.html``,
            ``.htm``, ``.epub``, ``.odt``).

    Returns:
        List of ``PageResult`` objects, one per page or logical unit.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path).resolve()
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS.keys()))
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported: {supported}"
        )

    if suffix == ".pdf":
        return _load_pdf(str(path))
    elif suffix in (".txt", ".md"):
        return _load_txt(str(path))
    else:
        handler = FORMAT_HANDLERS[suffix]
        return handler(str(path))


def _load_pdf(file_path: str) -> list[PageResult]:
    """Load a PDF via full-document OCR (300 DPI, Nepali Tesseract)."""
    return ocr_all_pages(file_path)


def _load_txt(file_path: str) -> list[PageResult]:
    """Load a plain-text or Markdown file as a single ``PageResult``."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    return [PageResult(
        page_number=1,
        text=text.strip(),
        source=str(path),
        extraction_method="text_file"
    )]


def main():
    """Load a document and write each page result to a JSON file in ingestion/tests/."""
    file_path = "data/sample_docs/test.pdf"
    results = load_document(file_path)

    # Resolve tests directory relative to this file (ingestion/tests/)
    tests_dir = Path(__file__).resolve().parent / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    for page_result in results:
        output_file = tests_dir / f"page_{page_result.page_number}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(page_result), f, ensure_ascii=False, indent=2)
        print(f"Wrote {output_file}")

    print(f"\nDone. {len(results)} page(s) written to {tests_dir}")


if __name__ == "__main__":
    main()
