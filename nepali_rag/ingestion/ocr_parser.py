import fitz
from PIL import Image
import pytesseract
import io
from pathlib import Path

from nepali_rag.config.settings import settings
from nepali_rag.ingestion.models import PageResult


def ocr_page(file_path: str, page_number: int) -> PageResult:
    """Run Tesseract OCR on a single page of a PDF at 300 DPI.

    page_number is 1-indexed. Returns a PageResult with extraction_method="ocr".
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
    """Run Tesseract OCR on every page of a PDF at 300 DPI.

    Returns a list of PageResult with extraction_method="ocr" for all pages.
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
