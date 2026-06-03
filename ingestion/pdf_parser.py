import fitz
from pathlib import Path

from ingestion.models import PageResult


def extract_digital_text(file_path: str) -> list[PageResult]:
    """Extract text from a PDF using PyMuPDF digital text extraction.

    Returns a list of PageResult for each page. Pages where digital extraction
    yields no text will have an empty string in the text field.
    """
    file_path = str(Path(file_path).resolve())
    results = []

    doc = fitz.open(file_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            results.append(PageResult(
                page_number=page_num + 1,
                text=text.strip() if text else "",
                source=file_path,
                extraction_method="digital"
            ))
    finally:
        doc.close()

    return results
