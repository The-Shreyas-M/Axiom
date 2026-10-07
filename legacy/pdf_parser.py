"""PDF parsing: extracts text, tables, figures/images, and detects scanned pages.

Uses PyMuPDF (fitz). Tables are flattened to text rows; figure images are
rendered as cropped PNGs with their captions when detectable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pymupdf as fitz  # PyMuPDF

MIN_CHARS_FOR_TEXT_PAGE = 60
SCANNED_IMAGE_COVERAGE = 0.5  # page is a scan when placed images cover this fraction of the page
MIN_FIGURE_SIZE = 80  # px, skip tiny images (logos, icons)
CAPTION_SEARCH_DIST = 80.0  # points below/above image to look for caption
FIGURE_OUTPUT_DIR = "output"

# Vector-drawn figures (no raster image embedded) are detected by clustering
# drawing paths. Thresholds are in points (1 pt = 1/72 inch).
MIN_VECTOR_FIGURE_W = 110.0
MIN_VECTOR_FIGURE_H = 80.0
MIN_VECTOR_FIGURE_AREA = 12000.0
MAX_VECTOR_PAGE_FRACTION = 0.6  # skip full-page "background" drawing clusters
VECTOR_CLUSTER_GAP = 10.0

_FIGURE_RE = re.compile(r"^(figure|fig\.?)\s*\d+", re.IGNORECASE)
_TABLE_RE = re.compile(r"^(table)\s*\d+", re.IGNORECASE)


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Table:
    number: int
    page: int
    text: str  # flattened, " | " separated rows


@dataclass
class Figure:
    number: int
    page: int
    image_path: str
    label: str = ""
    caption: str = ""


@dataclass
class Paper:
    path: str
    name: str
    pages: list[Page] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def stats(self) -> dict:
        return {
            "pages": len(self.pages),
            "tables": len(self.tables),
            "figures": len(self.figures),
            "scanned_pages": len(self.scanned_pages),
        }


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80]


def _page_image_coverage(page) -> float:
    """Fraction of the page area covered by placed images (scans, strip scans)."""
    page_area = max(page.rect.width * page.rect.height, 1.0)
    covered = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.is_empty or r.is_infinite:
            continue
        covered += r.get_area()
    return min(covered / page_area, 1.0)


def _is_scanned_page(page) -> bool:
    """True for scanned pages.

    A page is a scan when it has little extractable text, or when a scan image
    covers most of the page. Scans often carry a baked-in OCR text layer, so the
    text-length check alone misses them (e.g. image-strip scans where every
    page is one scan tiled into thin strips)."""
    text = page.get_text("text")
    if len(text.strip()) < MIN_CHARS_FOR_TEXT_PAGE:
        return True
    return _page_image_coverage(page) >= SCANNED_IMAGE_COVERAGE


def _flatten_table(table) -> str:
    rows = []
    for row in table.extract():
        cells = [str(c).strip().replace("\n", " ") if c is not None else "" for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _find_caption(page, image_rect, above: bool = False) -> str:
    """Look for a caption block (Figure/Table N: ...) near an image.

    Searches below the image by default, or above when ``above`` is set.
    """
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = block
        if above:
            if y1 > image_rect.y0 + 2:
                continue
            if image_rect.y0 - y1 > CAPTION_SEARCH_DIST:
                continue
        else:
            if y0 < image_rect.y1 - 2:
                continue
            if y0 - image_rect.y1 > CAPTION_SEARCH_DIST:
                continue
        if x0 > image_rect.x0 + 40 or x1 < image_rect.x0 - 40:
            continue
        clean = text.strip().replace("\n", " ")
        if _FIGURE_RE.match(clean) or _TABLE_RE.match(clean):
            return clean
    return ""


def _cluster_rects(rects: list, gap: float = VECTOR_CLUSTER_GAP) -> list[dict]:
    """Greedily merge overlapping/nearby rects into drawing clusters."""
    clusters = [{"bbox": r, "n": 1} for r in rects if not (r.is_empty or r.is_infinite)]
    merged = True
    while merged:
        merged = False
        new_clusters: list[dict] = []
        for c in clusters:
            placed = False
            for nc in new_clusters:
                b = nc["bbox"]
                bb = c["bbox"]
                if (
                    bb.x0 <= b.x1 + gap and bb.x1 >= b.x0 - gap
                    and bb.y0 <= b.y1 + gap and bb.y1 >= b.y0 - gap
                ):
                    nc["bbox"] = b | bb
                    nc["n"] += c["n"]
                    placed = True
                    merged = True
                    break
            if not placed:
                new_clusters.append(dict(c))
        clusters = new_clusters
    return clusters


def _extract_vector_figures(page, pno: int, fig_dir: str, fig_counter: int,
                            seen_image_labels: set, raster_rects: list,
                            paper: Paper) -> int:
    """Extract vector-drawn figures (no embedded raster image) as PNGs."""
    paths = page.get_drawings()
    if not paths:
        return fig_counter

    rects, n_filled = [], 0
    for p in paths:
        r = p.get("rect")
        if r is None:
            continue
        rects.append(r)
        if p.get("fill") is not None:
            n_filled += 1
    if len(rects) < 4:
        return fig_counter

    page_area = page.rect.width * page.rect.height
    for cl in _cluster_rects(rects):
        b = cl["bbox"]
        if b.width < MIN_VECTOR_FIGURE_W or b.height < MIN_VECTOR_FIGURE_H:
            continue
        area = b.width * b.height
        if area < MIN_VECTOR_FIGURE_AREA:
            continue
        if area > MAX_VECTOR_PAGE_FRACTION * page_area:
            continue  # likely full-page background/scan
        # skip clusters that are really an already-extracted raster figure
        if any((r & b).get_area() > 0.5 * b.get_area() for r in raster_rects):
            continue

        caption = _find_caption(page, b) or _find_caption(page, b, above=True)
        # only treat as a figure when we have a Figure caption OR strong
        # drawing evidence (filled shapes or many paths) — avoids table rules
        is_figure = bool(caption and _FIGURE_RE.match(caption))
        is_figure = is_figure or (n_filled >= 2 and area >= MIN_VECTOR_FIGURE_AREA)
        is_figure = is_figure or (cl["n"] >= 15 and area >= MIN_VECTOR_FIGURE_AREA)
        if not is_figure:
            continue

        label = ""
        if caption:
            m = re.match(r"^(figure|fig\.?)\s*(\d+)", caption, re.IGNORECASE)
            if m:
                label = f"{m.group(1).capitalize()} {m.group(2)}"
                if label.lower() in seen_image_labels:
                    continue
                seen_image_labels.add(label.lower())

        fig_counter += 1
        img_path = os.path.join(fig_dir, f"fig_{fig_counter:03d}_p{pno}.png")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=b)
        pix.save(img_path)
        paper.figures.append(
            Figure(number=fig_counter, page=pno, image_path=img_path, label=label, caption=caption)
        )
    return fig_counter


def parse_pdf(path: str, output_dir: str = FIGURE_OUTPUT_DIR) -> Paper:
    """Extract pages, tables and figures from a PDF."""
    doc = fitz.open(path)
    name = _safe_name(os.path.splitext(os.path.basename(path))[0])
    paper = Paper(path=path, name=name)

    fig_dir = os.path.join(output_dir, name, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig_counter = 0
    seen_image_labels: set[str] = set()

    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text")
        paper.pages.append(Page(number=pno, text=text))

        if _is_scanned_page(page):
            paper.scanned_pages.append(pno)

        # ---- tables ----
        for table in page.find_tables().tables:
            flat = _flatten_table(table)
            if len(flat) > 10:
                paper.tables.append(
                    Table(number=len(paper.tables) + 1, page=pno, text=flat)
                )

        # ---- figures (raster images placed on the page) ----
        raster_rects: list = []
        for xref in set(i[0] for i in page.get_images(full=True)):
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = max(rects, key=lambda r: r.width * r.height)
            if rect.width < MIN_FIGURE_SIZE or rect.height < MIN_FIGURE_SIZE:
                continue
            raster_rects.append(rect)

            caption = _find_caption(page, rect)
            label = ""
            if caption:
                m = re.match(r"^(figure|fig\.?|table)\s*(\d+)", caption, re.IGNORECASE)
                if m:
                    label = f"{m.group(1).capitalize()} {m.group(2)}"
                    if label.lower() in seen_image_labels:
                        continue  # logo/watermark repeated on every page
                    seen_image_labels.add(label.lower())

            fig_counter += 1
            img_path = os.path.join(fig_dir, f"fig_{fig_counter:03d}_p{pno}.png")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
            pix.save(img_path)

            paper.figures.append(
                Figure(number=fig_counter, page=pno, image_path=img_path, label=label, caption=caption)
            )

        # ---- figures (vector-drawn, no embedded raster image) ----
        fig_counter = _extract_vector_figures(
            page, pno, fig_dir, fig_counter, seen_image_labels, raster_rects, paper
        )

    doc.close()
    return paper
