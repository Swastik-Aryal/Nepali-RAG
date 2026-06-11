"""PDF OCR extraction using PyMuPDF rendering and Tesseract.

Pages are rasterized at 300 DPI before OCR with languages from
``settings.tesseract_languages`` (``nep``).
"""

import fitz
from PIL import Image
import pytesseract
import io
from pathlib import Path

from nepali_rag.config.settings import settings
from nepali_rag.ingestion.models import PageResult


def ocr_page(file_path: str, page_number: int) -> PageResult:
    """OCR a single PDF page at 300 DPI.

    Args:
        file_path: Path to the PDF file.
        page_number: 1-indexed page number.

    Returns:
        ``PageResult`` with ``extraction_method="ocr"``.
    """
    file_path = str(Path(file_path).resolve())

    doc = fitz.open(file_path)
    try:
        page = doc[page_number - 1]
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)

        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(
            image,
            lang=settings.tesseract_languages
        )
    finally:
        doc.close()

    return PageResult(
        page_number=page_number,
        text=text.strip() if text else "",
        source=file_path,
        extraction_method="ocr"
    )


def ocr_all_pages(file_path: str) -> list[PageResult]:
    """OCR every page of a PDF at 300 DPI.

    Returns:
        List of ``PageResult`` objects (one per page), each with ``extraction_method="ocr"``.
    """
    file_path = str(Path(file_path).resolve())
    results = []

    doc = fitz.open(file_path)
    try:
        mat = fitz.Matrix(300 / 72, 300 / 72)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)

            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            text = pytesseract.image_to_string(
                image,
                lang=settings.tesseract_languages
            )

            results.append(PageResult(
                page_number=page_num + 1,
                text=text.strip() if text else "",
                source=file_path,
                extraction_method="ocr"
            ))
    finally:
        doc.close()

    return results
