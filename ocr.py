"""OCR for scanned PDF pages.

Multi-engine facade: tries PaddleOCR (fast, multi-language, CPU-friendly)
first, then falls back to EasyOCR. Engines are detected via importlib (no
heavy import until actually used), so OCR is cleanly skipped when no engine
is installed.

Both engine APIs are handled:
  - paddleocr 2.x: predictor.ocr(img, cls=False) -> [[box, (text, conf)], ...]
  - paddleocr 3.x: predictor.predict(img) -> [OCRResult]; .json["res"][...]
  - easyocr:       reader.readtext(img, detail=1) -> [(box, text, conf), ...]

Recognized lines are clustered into reading-order rows (top-to-bottom,
left-to-right within a row band) so extracted text stays readable.
"""

from __future__ import annotations

import glob as _glob
import importlib.util
import os
import sys

import pymupdf as fitz


def _add_nvidia_paths() -> None:
    """Prepend pip-installed NVIDIA CUDA/cuDNN DLL dirs to PATH for paddle.

    paddlepaddle-gpu on Windows does not bundle CUDA/cuDNN runtimes, so we
    pull them from the pip packages (nvidia-cudnn-cu12, nvidia-cublas-cu12,
    nvidia-cuda-runtime-cu12) when present.
    """
    site = os.path.join(sys.prefix, "Lib", "site-packages")
    dirs = [d for d in _glob.glob(os.path.join(site, "nvidia", "*", "bin")) + _glob.glob(os.path.join(site, "nvidia", "*", "lib")) if os.path.isdir(d)]
    if not dirs:
        return
    path = os.environ.get("PATH", "")
    new = [d for d in dirs if d not in path]
    if new:
        os.environ["PATH"] = os.pathsep.join(new) + os.pathsep + path


class OCR:
    """Lazy OCR wrapper; reports which engine is available, if any."""

    def __init__(self):
        self.engine_name = self._detect_engine()
        self.available = self.engine_name != "none"
        self._paddle = None
        self._paddle3 = False
        self._easy = None

    @staticmethod
    def _detect_engine() -> str:
        if importlib.util.find_spec("paddleocr"):
            return "paddleocr"
        if importlib.util.find_spec("easyocr"):
            return "easyocr"
        return "none"

    # -- engine loading ----------------------------------------------------
    def _engine(self):
        if self.engine_name == "paddleocr":
            if self._paddle is None:
                os.environ.setdefault("PADDLE_PDX_LOGGER_LEVEL", "ERROR")
                _add_nvidia_paths()
                import paddleocr
                from paddleocr import PaddleOCR

                self._paddle3 = int(paddleocr.__version__.split(".")[0]) >= 3
                if self._paddle3:
                    self._paddle = PaddleOCR(
                        lang="en",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        enable_mkldnn=False,  # oneDNN inference path is buggy in paddlepaddle 3.3
                    )
                else:
                    self._paddle = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
            return self._paddle

        if self._easy is None:
            import easyocr

            self._easy = easyocr.Reader(["en"], gpu=False)
        return self._easy

    # -- public API --------------------------------------------------------
    def ocr_pages(self, doc_path: str, page_numbers: list[int], dpi: int = 200) -> dict[int, str]:
        """OCR the given 1-based page numbers; returns {page_num: text}."""
        if not self.available or not page_numbers:
            return {}
        engine = self._engine()
        results: dict[int, str] = {}
        doc = fitz.open(doc_path)
        try:
            for pno in page_numbers:
                page = doc[pno - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                img = _pix_to_numpy(pix)
                lines = self._read(engine, img)
                text = _order_lines(lines)
                if text:
                    results[pno] = text
        finally:
            doc.close()
        return results

    # -- engine dispatch ---------------------------------------------------
    def _read(self, engine, img) -> list[tuple[list, str]]:
        if self.engine_name == "paddleocr":
            # PaddleOCR expects BGR input arrays.
            return _parse_paddle(engine, img[:, :, ::-1].copy(), self._paddle3)
        return _parse_easy(engine, img)


def _pix_to_numpy(pix):
    import numpy as np

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    elif pix.n == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def _parse_paddle(engine, img, is_paddle3: bool) -> list[tuple[list, str]]:
    if not is_paddle3:
        res = engine.ocr(img, cls=False)  # paddleocr 2.x
    else:
        res = list(engine.predict(img))  # paddleocr 3.x

    if not res:
        return []

    if is_paddle3:
        lines: list[tuple[list, str]] = []
        for page_res in res:
            try:
                items = page_res.json.get("res", [])
            except AttributeError:
                continue
            for item in items:
                box = item.get("box")
                text = item.get("rec_text")
                if box and text:
                    lines.append((box, str(text)))
        return lines

    # 2.x: engine.ocr(img) returns [[box, (text, conf)], ...] wrapped in an
    # outer list when a single image is passed.
    inner = res[0] if (isinstance(res[0], list) and res[0] and isinstance(res[0][0], list)) else res
    lines = []
    for item in inner:
        if not item or len(item) < 2:
            continue
        box = item[0]
        entry = item[1]
        text = str(entry[0]) if isinstance(entry, (list, tuple)) else str(entry)
        if box and text.strip():
            lines.append((box, text))
    return lines


def _parse_easy(engine, img) -> list[tuple[list, str]]:
    lines = []
    for box, text, _conf in engine.readtext(img, detail=1):
        if box and text.strip():
            lines.append((box, text))
    return lines


def _order_lines(lines: list[tuple[list, str]], row_gap_ratio: float = 0.4, min_gap: float = 4.0) -> str:
    """Cluster OCR lines into reading-order rows."""
    rows = []
    for box, text in lines:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        if not xs or not ys:
            continue
        rows.append({"x": min(xs), "y": sum(ys) / len(ys), "h": max(ys) - min(ys), "text": text.strip()})
    if not rows:
        return ""

    rows.sort(key=lambda r: r["y"])
    grouped: list[dict] = []
    for r in rows:
        if grouped and (r["y"] - grouped[-1]["y"]) < (max(grouped[-1]["h"], 1.0) * row_gap_ratio + min_gap):
            prev = grouped[-1]
            prev["text"] = (prev["text"] + " " + r["text"]).strip()
            prev["y"] = (prev["y"] + r["y"]) / 2
            prev["h"] = max(prev["h"], r["h"])
            prev["x"] = min(prev["x"], r["x"])
        else:
            grouped.append(dict(r))

    grouped.sort(key=lambda g: (round(g["y"]), g["x"]))
    return "\n".join(g["text"] for g in grouped)
