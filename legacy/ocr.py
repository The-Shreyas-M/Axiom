"""OCR: EasyOCR on GPU for pages that need it.

PyMuPDF extracts text from most pages instantly. Only pages with
little/no extractable text (scans, images) go through EasyOCR.
This keeps processing fast while ensuring nothing is missed.
"""

from __future__ import annotations

import numpy as np
import pymupdf as fitz

MIN_CHARS = 60
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def _needs_ocr(page) -> bool:
    text = page.get_text("text").strip()
    if len(text) < MIN_CHARS:
        return True
    page_area = max(page.rect.width * page.rect.height, 1.0)
    covered = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            covered += r.get_area()
    return covered / page_area >= 0.5


def ocr_pages(doc_path: str, page_numbers: list[int], dpi: int = 150) -> dict[int, str]:
    """OCR only pages that need it. Returns {page_num: text}."""
    if not page_numbers:
        return {}

    reader = _get_reader()
    doc = fitz.open(doc_path)
    results: dict[int, str] = {}
    try:
        for pno in page_numbers:
            page = doc[pno - 1]
            if not _needs_ocr(page):
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n >= 4:
                img = img[:, :, :3]
            elif pix.n == 1:
                img = np.repeat(img, 3, axis=2)

            ocr_result = reader.readtext(img, detail=0, paragraph=True)
            text = "\n".join(t.strip() for t in ocr_result if t.strip())
            if text:
                results[pno] = text
    finally:
        doc.close()
    return results
