"""GPU OCR for every page using EasyOCR (PyTorch CUDA)."""

from __future__ import annotations

import numpy as np
import pymupdf as fitz

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def ocr_page(doc_path: str, page_number: int, dpi: int = 200) -> str:
    """OCR a single 1-based page number. Returns text."""
    reader = _get_reader()
    doc = fitz.open(doc_path)
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = img[:, :, :3]
        elif pix.n == 1:
            img = np.repeat(img, 3, axis=2)

        results = reader.readtext(img, detail=1, paragraph=True)
        lines = []
        for item in results:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text and text.strip():
                    lines.append(text.strip())
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[1]) if not isinstance(item[1], (list, tuple)) else ""
                if text and text.strip():
                    lines.append(text.strip())
            elif isinstance(item, str) and item.strip():
                lines.append(item.strip())
        return "\n".join(lines)
    finally:
        doc.close()


def ocr_all_pages(doc_path: str, page_numbers: list[int], dpi: int = 200) -> dict[int, str]:
    """OCR multiple pages. Returns {page_num: text}."""
    results: dict[int, str] = {}
    for pno in page_numbers:
        text = ocr_page(doc_path, pno, dpi)
        if text.strip():
            results[pno] = text
    return results
