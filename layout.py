"""Layout analysis for scanned pages using PaddleOCR PP-Structure.

Scanned pages are single images, so PyMuPDF can find no tables, figures or
structure in them. PP-Structure detects layout regions (text / figure /
table / formula) directly on the rendered page image, which lets the pipeline
extract figures and tables that are otherwise invisible to structure-based
parsing. Runs on GPU via the same paddle runtime as the OCR engine.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from html import unescape

import numpy as np
import pymupdf as fitz

import ocr
from pdf_parser import FIGURE_OUTPUT_DIR, Figure, Paper, Table

DPI = 150
SCALE = DPI / 72.0

MIN_FIGURE_W = 60.0
MIN_FIGURE_H = 40.0
MAX_FIGURE_FRACTION = 0.7  # skip regions that cover nearly the whole page
MIN_SCORE = 0.45
CAPTION_DIST = 45.0  # points, how far a caption may sit from its figure

_CAPTION_RE = re.compile(r"^(figure|fig\.?)\s*\d+", re.IGNORECASE)
_TABLE_RE = re.compile(r"^(table)\s*\d+", re.IGNORECASE)


def _caption_head(text: str) -> str:
    """Lowercased alnum head; OCR noise 'F1G' -> 'fig', keeping real digits."""
    head = re.sub(r"[^a-z0-9]", "", text[:12].lower())
    return re.sub(r"f1(ure|g|a|c)", r"fi\1", head)


def _is_figure_caption(text: str) -> bool:
    """Loose match so OCR noise like 'Fia. 3(a)', 'FIg. 1.' or 'F1G.7' counts."""
    return re.match(r"^(?:figure|fig|f[a-z]{2})\d", _caption_head(text)) is not None


def _is_table_caption(text: str) -> bool:
    head = _caption_head(text)
    return head.startswith("table") and head[5:6].isdigit()


def _caption_label(caption: str) -> str:
    """'FIG. 3(a). Unnormalized set' -> label like 'Figure 3' (OCR-tolerant)."""
    head = _caption_head(caption[:24])
    m = re.search(r"(figure|fig|f[a-z]{2})\.?\(?(\d+)", head)
    if not m:
        return ""
    kw, num = m.group(1), m.group(2)
    if kw.startswith("figure"):
        return f"Figure {num}"
    if kw.startswith("table"):
        return f"Table {num}"
    return f"Fig. {num}"


@dataclass
class Region:
    kind: str  # 'figure' | 'table'
    bbox: tuple  # (x0, y0, x1, y1) in page points
    score: float = 0.0
    text: str = ""  # figure: inner OCR text; table: flattened cell text
    caption: str = ""  # matched "Figure N: ..." caption when found


@dataclass
class PageLayout:
    pno: int
    text: str = ""
    regions: list[Region] = field(default_factory=list)
    captions: list[tuple] = field(default_factory=list)  # (x0, y0, x1, y1, text)


def _html_to_text(html: str) -> str:
    """Flatten a PP-Structure table HTML string into " | " separated rows."""
    if not html:
        return ""
    html = re.sub(r"<br\s*/?>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"</tr>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</td>", " | ", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    rows = []
    for line in html.splitlines():
        clean = re.sub(r"\s+", " ", unescape(line)).strip().strip("|").strip()
        if clean:
            rows.append(clean)
    return "\n".join(rows)


class LayoutAnalyzer:
    """Lazy PP-Structure wrapper producing per-page layout for scanned pages."""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            ocr._add_nvidia_paths()
            from paddleocr import PPStructure

            self._engine = PPStructure(
                lang="en", show_log=False, use_gpu=True,
                use_fc=False, use_tdo=False,
            )
        return self._engine

    @property
    def available(self) -> bool:
        try:
            self._get_engine()
            return True
        except Exception:
            return False

    def analyze(self, doc_path: str, page_numbers: list[int]) -> dict[int, PageLayout]:
        """Run layout analysis on 1-based page numbers; {page_num: PageLayout}."""
        engine = self._get_engine()
        doc = fitz.open(doc_path)
        layouts: dict[int, PageLayout] = {}
        try:
            for pno in page_numbers:
                page = doc[pno - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )[:, :, :3].copy()
                layouts[pno] = self._parse(engine(img), pno)
        finally:
            doc.close()
        return layouts

    def _parse(self, result: list, pno: int) -> PageLayout:
        lines: list[tuple[list, str]] = []
        text_blocks: list[tuple] = []  # (x0, y0, x1, y1, text) for captions
        regions: list[Region] = []
        seen_lines: set[tuple] = set()

        for r in result:
            kind = r.get("type")
            bbox = r.get("bbox")
            res = r.get("res")
            score = float(r.get("score", 0.0))
            if not bbox:
                continue
            b = tuple(v / SCALE for v in bbox)

            if kind in ("text", "list"):
                joined = ""
                for item in res or []:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", ""))
                    if not text.strip():
                        continue
                    joined = f"{joined} {text}".strip()
                    poly = item.get("text_region")
                    if isinstance(poly, (list, tuple)) and len(poly) >= 4:
                        ys = [p[1] for p in poly]
                        xs = [p[0] for p in poly]
                        key = (round((min(ys) + max(ys)) / 2 / 2) * 2, text)
                        if key in seen_lines:
                            continue  # overlapping regions can repeat a line
                        seen_lines.add(key)
                        lines.append((poly, text))
                if joined.strip():
                    text_blocks.append((b[0], b[1], b[2], b[3], joined.strip()))
            elif kind == "figure":
                inner = [str(x.get("text", "")) for x in (res or []) if isinstance(x, dict)]
                regions.append(Region(kind="figure", bbox=b, score=score, text=" ".join(inner)))
            elif kind == "table":
                html = res.get("html", "") if isinstance(res, dict) else ""
                regions.append(
                    Region(kind="table", bbox=b, score=score, text=_html_to_text(html))
                )

        layout = PageLayout(pno=pno, text=ocr._order_lines(lines), regions=regions)
        for region in layout.regions:
            if region.kind == "figure":
                region.caption = self._find_caption(region.bbox, text_blocks)
        layout.captions = [
            (b[0], b[1], b[2], b[3], b[4])
            for b in text_blocks
            if _is_figure_caption(b[4]) or _is_table_caption(b[4])
        ]
        return layout

    def _find_caption(self, bbox: tuple, text_blocks: list[tuple]) -> str:
        """Nearest "Figure/Table N: ..." block below, else above the figure."""
        x0, y0, x1, y1 = bbox
        below = above = None
        for bx0, by0, bx1, by1, text in text_blocks:
            if not (_is_figure_caption(text) or _is_table_caption(text)):
                continue
            if bx1 < x0 - 40 or bx0 > x1 + 40:
                continue
            if by0 >= y1 - 6 and by0 - y1 <= CAPTION_DIST:
                if below is None or by0 < below[0]:
                    below = (by0, text)
            elif by1 <= y0 + 6 and y0 - by1 <= CAPTION_DIST:
                if above is None or by1 > above[0]:
                    above = (by1, text)
        if below:
            return below[1]
        if above:
            return above[1]
        return ""


def _caption_figure_rect(cap: tuple, regions: list[Region],
                         captions: list[tuple], page_rect) -> tuple | None:
    """Recover a figure PP-Structure missed: the band above its caption.

    Returns (x0, y0, x1, y1) in page points, or None when the band is not
    well-bounded (no anchor above, or overlaps an already-detected figure).
    """
    cx0, cy0, cx1, cy1, _ = cap
    page_h = page_rect.height
    anchors = [tuple(r.bbox) for r in regions] + [tuple(c[:4]) for c in captions]
    anchors = [
        a for a in anchors
        if a[3] <= cy0 and not (a[0] < cx1 + 10 and a[2] > cx0 - 10 and a[3] > cy0 - 2)
    ]
    if not anchors:
        return None
    top = max(a[3] for a in anchors)
    if top < cy0 - 0.45 * page_h:
        return None  # figure would start near the top of the page; do not guess
    bottom = cy0 - 3.0
    if bottom - top < 25.0 or bottom - top > 0.5 * page_h:
        return None

    band = [tuple(r.bbox) for r in regions] + [tuple(c[:4]) for c in captions]
    left = right = None
    for b in band:
        if b[3] < top or b[1] > bottom:
            continue
        if b[2] < cx0 - 15:
            left = b[2] if left is None else max(left, b[2])
        if b[0] > cx1 + 15:
            right = b[0] if right is None else min(right, b[0])
    x0 = (left + 6) if left is not None else page_rect.x0 + 8
    x1 = (right - 6) if right is not None else page_rect.x1 - 8
    if x1 - x0 < 40:
        return None

    rect = fitz.Rect(x0, top, x1, bottom)
    for r in regions:
        if r.kind != "figure":
            continue
        other = fitz.Rect(*r.bbox)
        inter = rect & other
        if not inter.is_empty and inter.get_area() > 0.3 * other.get_area():
            return None  # already covered by a detected figure region
    return (x0, top, x1, bottom)


def _save_figure(paper: Paper, pno: int, doc, fig_dir: str,
                 rect: tuple, caption: str) -> None:
    num = len(paper.figures) + 1
    img_path = os.path.join(fig_dir, f"fig_{num:03d}_p{pno}.png")
    page = doc[pno - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(*rect))
    pix.save(img_path)
    paper.figures.append(
        Figure(number=num, page=pno, image_path=img_path,
               label=_caption_label(caption), caption=caption or "")
    )


def apply_layout(paper: Paper, layouts: dict[int, PageLayout]) -> list[int]:
    """Merge layout results into a parsed Paper (page text, figures, tables).

    Returns the page numbers whose text was replaced by layout OCR text.
    """
    fig_dir = os.path.join(FIGURE_OUTPUT_DIR, paper.name, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    doc = fitz.open(paper.path)
    ocr_pages: list[int] = []
    try:
        for pno, layout in layouts.items():
            if layout.text:
                ocr_pages.append(pno)
                paper.pages[pno - 1].text = layout.text

            page = doc[pno - 1]
            page_area = page.rect.width * page.rect.height
            for region in layout.regions:
                if region.kind == "figure":
                    if not _valid_figure(region, page_area):
                        continue
                    caption = region.caption or region.text or ""
                    _save_figure(paper, pno, doc, fig_dir, region.bbox, caption)
                elif region.kind == "table":
                    if len(region.text) > 10:
                        paper.tables.append(
                            Table(number=len(paper.tables) + 1, page=pno, text=region.text)
                        )

            recovered: list[tuple] = []
            for cap in layout.captions:
                rect = _caption_figure_rect(cap, layout.regions, layout.captions, page.rect)
                if rect is None:
                    continue
                if any(not (fitz.Rect(*rect) & fitz.Rect(*pr)).is_empty for pr in recovered):
                    continue
                recovered.append(rect)
                _save_figure(paper, pno, doc, fig_dir, rect, cap[4])
    finally:
        doc.close()
    return ocr_pages


def _valid_figure(region: Region, page_area: float) -> bool:
    x0, y0, x1, y1 = region.bbox
    w, h = x1 - x0, y1 - y0
    if w < MIN_FIGURE_W or h < MIN_FIGURE_H:
        return False
    if region.score < MIN_SCORE:
        return False
    if w * h > MAX_FIGURE_FRACTION * page_area:
        return False
    return True
