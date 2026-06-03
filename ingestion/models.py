from dataclasses import dataclass


@dataclass
class PageResult:
    page_number: int
    text: str
    source: str
    extraction_method: str
