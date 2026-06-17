"""Shared data models for the ingestion pipeline."""

from dataclasses import dataclass
import os


@dataclass
class PageResult:
    """Text extracted from a single page (or logical unit) of a source document.

    Attributes:
        page_number: 1-indexed page or slide/sheet/chapter number.
        text: Raw extracted text for that page.
        source: Absolute path to the source file.
        extraction_method: How text was obtained (e.g. ``ocr``, ``docx``, ``text_file``).
    """

    page_number: int
    text: str
    source: str
    extraction_method: str

    def __post_init__(self):
        self.source = os.path.basename(self.source)
