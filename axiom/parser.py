"""PDF -> list of elements (text chunks, figures, tables, captions), digital or scanned.

Element: {id, type: text|figure|table|caption, page, bbox, section, text, caption, image, mentions, desc}
"""
import re
from collections import Counter
from pathlib import Path

import cv2
import fitz
import numpy as np

CAP_RE = re.compile(r"^\s*(figure|fig\.?|table|tab\.?)\s*([A-Z]?\d+[a-z]?)\s*[:.\-–—|]\s*\S", re.I)
HEAD_RE = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.)?\s*[A-Z][^.!?]{1,90}$")
KNOWN_HEADS = {"abstract", "introduction", "related work", "background", "methods", "method", "methodology",
               "experiments", "results", "discussion", "conclusion", "conclusions", "references",
               "acknowledgements", "acknowledgments", "appendix", "limitations", "evaluation"}
ZOOM = 2.0
CHUNK = 900

_ocr = None


def ocr_engine():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def cap_key(text):
    m = CAP_RE.match(text)
    if not m:
        return None
    kind = "table" if m.group(1).lower().startswith("tab") else "figure"
    return kind, m.group(2).upper()


# ---------------------------------------------------------------- digital pages
def _blocks(page):
    """Text blocks with font stats: (bbox, text, size, bold)."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        txt, sizes, bold, n = [], 0.0, 0, 0
        for ln in b["lines"]:
            s = "".join(sp["text"] for sp in ln["spans"]).strip()
            if s:
                txt.append(s)
            for sp in ln["spans"]:
                L = len(sp["text"])
                sizes += sp["size"] * L
                n += L
                bold += L if (sp["flags"] & 16 or "bold" in sp["font"].lower()) else 0
        t = " ".join(txt)
        t = re.sub(r"-\s(?=[a-z])", "", t)  # de-hyphenate wrapped words
        if t.strip() and n:
            out.append({"bbox": tuple(b["bbox"]), "text": t.strip(), "size": sizes / n,
                        "bold": bold / n > 0.6, "lines": len(b["lines"])})
    return out


def _reading_order(blocks, W):
    """Handle 2-column layouts: full-width blocks split page into bands; within band left col then right."""
    full = [b for b in blocks if b["bbox"][2] - b["bbox"][0] > 0.6 * W]
    cols = [b for b in blocks if b not in full]
    full.sort(key=lambda b: b["bbox"][1])
    ordered, cur = [], 0.0
    bounds = [(f["bbox"][1], f) for f in full]
    bands = []
    prev = -1
    for y, f in bounds:
        bands.append((prev, y, f))
        prev = y
    bands.append((prev, 1e9, None))
    used = set()
    for lo, hi, f in bands:
        seg = [b for b in cols if lo <= b["bbox"][1] < hi and id(b) not in used]
        used.update(id(b) for b in seg)
        left = sorted([b for b in seg if (b["bbox"][0] + b["bbox"][2]) / 2 < W / 2], key=lambda b: b["bbox"][1])
        right = sorted([b for b in seg if (b["bbox"][0] + b["bbox"][2]) / 2 >= W / 2], key=lambda b: b["bbox"][1])
        ordered += left + right
        if f:
            ordered.append(f)
    return ordered


def _inside(b, r, tol=2):
    return b[0] >= r[0] - tol and b[1] >= r[1] - tol and b[2] <= r[2] + tol and b[3] <= r[3] + tol


def _digital_regions(page, want_tables=True):
    """Figures (raster + vector) and tables as bboxes."""
    W, H = page.rect.width, page.rect.height
    tables = []
    try:
        for t in (page.find_tables().tables if want_tables else []):
            if t.row_count >= 2 and t.col_count >= 2:
                tables.append((tuple(t.bbox), t))
    except Exception:
        pass
    figs = []
    for im in page.get_image_info():
        r = fitz.Rect(im["bbox"]) & page.rect
        if r.width > 60 and r.height > 50:
            figs.append(tuple(r))
    try:
        for r in page.cluster_drawings():
            r = fitz.Rect(r) & page.rect
            if r.width > 90 and r.height > 70 and r.width * r.height < 0.9 * W * H:
                figs.append(tuple(r))
    except Exception:
        pass
    figs = _merge_rects(figs)
    figs = [f for f in figs if not any(_overlap(f, t[0]) > 0.5 for t in tables)]
    return figs, tables


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0


def _merge_rects(rects, gap=12):
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    rects.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [tuple(r) for r in rects]


# ---------------------------------------------------------------- scanned pages
def _ocr_page(page, img_bgr):
    """OCR a rendered page -> blocks in PDF-point coords, plus raw word boxes (pixel coords)."""
    res, _ = ocr_engine()(img_bgr)
    lines = []
    for box, text, conf in res or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append({"px": (min(xs), min(ys), max(xs), max(ys)), "text": text.strip(), "conf": float(conf)})
    return [l for l in lines if l["text"]]


def _group_lines(lines, scale):
    """Merge OCR lines into paragraph blocks (column-aware via x-overlap + vertical gap)."""
    lines = sorted(lines, key=lambda l: (l["px"][1], l["px"][0]))
    blocks = []
    for l in lines:
        x0, y0, x1, y1 = l["px"]
        h = y1 - y0
        placed = False
        for b in ([] if cap_key(l["text"]) else reversed(blocks[-12:])):
            bx0, by0, bx1, by1 = b["px"]
            xo = min(x1, bx1) - max(x0, bx0)
            if xo > 0.5 * min(x1 - x0, bx1 - bx0) and 0 <= y0 - by1 < 0.9 * h and abs(h - b["h"]) < 0.5 * h:
                b["texts"].append(l["text"])
                b["px"] = (min(x0, bx0), by0, max(x1, bx1), y1)
                placed = True
                break
        if not placed:
            blocks.append({"px": (x0, y0, x1, y1), "texts": [l["text"]], "h": h})
    out = []
    for b in blocks:
        x0, y0, x1, y1 = b["px"]
        out.append({"bbox": (x0 / scale, y0 / scale, x1 / scale, y1 / scale), "text": " ".join(b["texts"]),
                    "size": b["h"] / scale * 0.8, "bold": False, "lines": len(b["texts"])})
    return out


def _scan_regions(img_bgr, lines, scale):
    """Find figures/tables on a scanned page from non-text ink + ruling lines."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    # tables: long horizontal + vertical rules
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 40)))
    grid = cv2.add(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    grid = cv2.dilate(grid, np.ones((15, 15), np.uint8))
    tables = []
    for c in cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 0.3 * w and ch > 0.04 * h:
            tables.append((x / scale, y / scale, (x + cw) / scale, (y + ch) / scale))
    # figures: ink left after blanking text boxes, dilated into blobs
    ink = bw.copy()
    for l in lines:
        x0, y0, x1, y1 = [int(v) for v in l["px"]]
        ink[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 0
    ink = cv2.dilate(ink, np.ones((25, 25), np.uint8))
    figs = []
    for c in cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 0.15 * w and ch > 0.08 * h and cv2.contourArea(c) > 0.02 * w * h:
            figs.append((x / scale, y / scale, (x + cw) / scale, (y + ch) / scale))
    figs = [f for f in _merge_rects(figs, 20) if not any(_overlap(f, t) > 0.4 for t in tables)]
    return figs, tables


def _rows_to_md(lines_in_region):
    """Rebuild a table from OCR boxes: cluster by row (y), order cells by x."""
    ls = sorted(lines_in_region, key=lambda l: (l["px"][1] + l["px"][3]) / 2)
    rows, cur, cy = [], [], None
    for l in ls:
        y = (l["px"][1] + l["px"][3]) / 2
        hh = l["px"][3] - l["px"][1]
        if cy is None or abs(y - cy) < 0.6 * hh:
            cur.append(l)
            cy = y if cy is None else (cy + y) / 2
        else:
            rows.append(cur)
            cur, cy = [l], y
    if cur:
        rows.append(cur)
    md = []
    for r in rows:
        md.append("| " + " | ".join(c["text"] for c in sorted(r, key=lambda l: l["px"][0])) + " |")
    return "\n".join(md)


# ---------------------------------------------------------------- main
def _crop(page, rect, path, pad=4):
    r = fitz.Rect(rect) + (-pad, -pad, pad, pad)
    r &= page.rect
    page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=r).save(path)


def parse_pdf(pdf_path, out_dir, progress=lambda p, msg: None):
    out_dir = Path(out_dir)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    n = len(doc)
    elements, counters = [], Counter()
    # body font size = most common by char count (for heading detection)
    sizes = Counter()
    for p in doc:
        for b in _blocks(p):
            sizes[round(b["size"])] += len(b["text"])
    body = sizes.most_common(1)[0][0] if sizes else 10
    stats = {"pages": n, "scanned_pages": 0}

    def new(kind, page, bbox, **kw):
        counters[kind] += 1
        e = {"id": f"{ {'figure':'fig','table':'tab','caption':'cap'}.get(kind,'p') }{counters['all'] + 1}", "type": kind, "page": page + 1,
             "bbox": [round(v, 1) for v in bbox], "section": "", "text": "", "caption": "", "image": None,
             "mentions": [], "desc": ""}
        counters["all"] += 1
        e.update(kw)
        return e

    section = "Front matter"
    for pno, page in enumerate(doc):
        progress(pno / n, f"Parsing page {pno + 1}/{n}")
        W = page.rect.width
        blocks = _blocks(page)
        chars = sum(len(b["text"]) for b in blocks)
        scale = 1.0
        scanned = chars < 80
        ocr_lines = []
        img_bgr = None
        if scanned:
            stats["scanned_pages"] += 1
            scale = 2.0
            pm = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img_bgr = cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3],
                                   cv2.COLOR_RGB2BGR)
            ocr_lines = _ocr_page(page, img_bgr)
            blocks = _group_lines(ocr_lines, scale)
            figs, tabs = _scan_regions(img_bgr, ocr_lines, scale)
            tables = [(t, None) for t in tabs]
        else:
            # find_tables is ~90% of parse time: only run it on pages that actually caption a table
            has_tcap = any((cap_key(b["text"]) or ("",))[0] == "table" for b in blocks)
            figs, tables = _digital_regions(page, want_tables=has_tcap)

        # figures / tables as elements
        page_el = []
        for i, f in enumerate(figs):
            if f[3] - f[1] < 40:
                continue
            e = new("figure", pno, f)
            page_el.append(e)
        for tb, tobj in tables:
            e = new("table", pno, tb)
            if tobj is not None:
                try:
                    e["text"] = tobj.to_markdown()
                except Exception:
                    e["text"] = "\n".join(" | ".join(str(c or "") for c in r) for r in tobj.extract())
            else:
                inside = [l for l in ocr_lines
                          if _inside(tuple(v / scale for v in l["px"]), tb, 3)]
                e["text"] = _rows_to_md(inside)
            page_el.append(e)

        # classify text blocks: in-figure/table text, captions, headings, body
        ordered = _reading_order(blocks, W)
        body_blocks = []
        for b in ordered:
            host = next((e for e in page_el if _inside(b["bbox"], e["bbox"], 4)), None)
            if host is not None:
                if host["type"] == "figure":
                    host["text"] = (host["text"] + " " + b["text"]).strip()
                continue
            ck = cap_key(b["text"])
            if ck and len(b["text"]) < 1500 and (b["lines"] <= 8):
                body_blocks.append(("caption", b, ck))
                continue
            t = b["text"]
            lower = re.sub(r"^[\dIVX.\s]+", "", t).strip().lower()
            is_head = (b["lines"] <= 2 and len(t) < 100 and not t.endswith(".") and
                       (lower in KNOWN_HEADS or ((b["bold"] or b["size"] > body * 1.08) and HEAD_RE.match(t))))
            if is_head:
                body_blocks.append(("head", b, None))
            elif re.fullmatch(r"\d{1,4}", t) or len(t) < 3:
                continue
            else:
                body_blocks.append(("text", b, None))

        # attach captions to nearest figure (caption below) / table (caption above)
        for kind, b, ck in body_blocks:
            if kind != "caption":
                continue
            ctype, num = ck
            cands = [e for e in page_el if e["type"] == ctype and not e["caption"]]
            if cands:
                def dist(e):
                    eb = e["bbox"]
                    below = b["bbox"][1] - eb[3]
                    above = eb[1] - b["bbox"][3]
                    d = min((x for x in (below, above) if x > -10), default=1e6)
                    return d if _overlap_x(eb, b["bbox"]) else d + 500
                tgt = min(cands, key=dist)
                if dist(tgt) < 400:
                    tgt["caption"], tgt["label"] = b["text"], f"{ctype.title()} {num}"
            elif ctype == "table":
                pass
        # merge uncaptioned figure fragments into an adjacent captioned figure (sub-panels), drop junk tables
        for e in [x for x in page_el if x["type"] == "figure" and not x["caption"]]:
            host = next((h for h in page_el if h["type"] == "figure" and h["caption"] and
                         e["bbox"][1] <= h["bbox"][3] + 20 and h["bbox"][1] <= e["bbox"][3] + 20 and
                         e["bbox"][0] - 200 <= h["bbox"][2] and h["bbox"][0] - 200 <= e["bbox"][2]), None)
            if host:
                host["bbox"] = [min(host["bbox"][0], e["bbox"][0]), min(host["bbox"][1], e["bbox"][1]),
                                max(host["bbox"][2], e["bbox"][2]), max(host["bbox"][3], e["bbox"][3])]
                page_el.remove(e)
        for e in [x for x in page_el if x["type"] == "table" and not x["caption"]]:
            rows = [r for r in e["text"].splitlines() if r.strip().startswith("|")]
            head = rows[0] if rows else ""
            junk = head.count("|Col") + head.count("Col") > 3
            if len(rows) < 4 or (e["text"].count("|") / max(1, len(rows))) < 4 or junk or (sum(len(c) for c in e["text"].replace("|", " ").split()) / max(1, len(e["text"].replace("|", " ").split()))) < 2.5:
                page_el.remove(e)
        # captioned table that find_tables missed: rebuild from the text blocks right under/over the caption
        for kind, b, ck in body_blocks:
            if kind == "caption" and ck[0] == "table" and not any(e["caption"] == b["text"] for e in page_el):
                below = sorted([x for x in body_blocks if x[0] == "text" and x[1]["bbox"][1] >= b["bbox"][3] - 2
                                and _overlap_x(x[1]["bbox"], b["bbox"])], key=lambda x: x[1]["bbox"][1])
                grab, limit = [], min(b["bbox"][3] + 260, page.rect.height - 20)
                for x in below:
                    if (len(x[1]["text"]) > 220 and x[1]["text"].count(". ") >= 2) or x[1]["bbox"][1] > limit:
                        limit = min(limit, x[1]["bbox"][1] - 3)
                        break
                    grab.append(x)
                rects = [tuple(x[1]["bbox"]) for x in grab]
                if not scanned:
                    try:
                        rects += [tuple(d["rect"]) for d in page.get_drawings()
                                  if b["bbox"][3] - 2 <= d["rect"].y0 and d["rect"].y1 <= limit
                                  and _overlap_x(tuple(d["rect"]), b["bbox"])]
                    except Exception:
                        pass
                if rects:
                    bb = (min(r[0] for r in rects), min(r[1] for r in rects),
                          max(r[2] for r in rects), max(r[3] for r in rects))
                    if bb[3] - bb[1] > 15:
                        e = new("table", pno, bb, caption=b["text"], label=f"Table {ck[1]}",
                                text=chr(10).join(x[1]["text"] for x in grab))
                        page_el.append(e)
                        body_blocks = [x for x in body_blocks if not any(x is g for g in grab)]
        # unmatched captions: figure-less caption (e.g. figure not detected) -> still a standalone element
        for kind, b, ck in body_blocks:
            if kind == "head":
                section = b["text"].strip()
            elif kind == "caption":
                owner = next((e for e in page_el if e["caption"] == b["text"]), None)
                label = f"{ck[0].title()} {ck[1]}"
                ce = new("caption", pno, b["bbox"], text=b["text"], section=section, label=label,
                         owner=owner["id"] if owner else None)
                elements.append(ce)
            else:
                elements.append(new("_para", pno, b["bbox"], text=b["text"], section=section))
        for e in page_el:
            e["section"] = section
            e.setdefault("label", "")
            if scanned:
                _crop_scan(img_bgr, e, scale, img_dir)
            else:
                path = img_dir / f"{e['id']}.png"
                _crop(page, e["bbox"], path)
                e["image"] = f"img/{path.name}"
            elements.append(e)
    doc.close()

    elements = _chunk_paragraphs(elements, counters)
    _link_mentions(elements)
    # reading order: by page then keep insertion order
    stats["counts"] = dict(Counter(e["type"] for e in elements))
    return elements, stats


def _overlap_x(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.3 * min(a[2] - a[0], b[2] - b[0])


def _crop_scan(img, e, scale, img_dir):
    x0, y0, x1, y1 = [int(v * scale) for v in e["bbox"]]
    h, w = img.shape[:2]
    crop = img[max(0, y0 - 8):min(h, y1 + 8), max(0, x0 - 8):min(w, x1 + 8)]
    path = img_dir / f"{e['id']}.png"
    cv2.imwrite(str(path), crop)
    e["image"] = f"img/{path.name}"


def _chunk_paragraphs(elements, counters):
    """Merge consecutive paragraphs within a section into ~CHUNK-char text elements (sentence-safe)."""
    out, buf, cur = [], [], None

    def flush():
        nonlocal buf
        if not buf:
            return
        counters["text"] += 1
        counters["all"] += 1
        out.append({"id": f"txt{counters['all']}", "type": "text", "page": buf[0]["page"],
                    "bbox": buf[0]["bbox"], "section": buf[0]["section"],
                    "text": " ".join(b["text"] for b in buf), "caption": "", "image": None,
                    "mentions": [], "desc": "", "pages": sorted({b["page"] for b in buf})})
        buf = []

    size = 0
    for e in elements:
        if e["type"] != "_para":
            out.append(e)
            continue
        if buf and (e["section"] != buf[0]["section"] or size + len(e["text"]) > CHUNK):
            flush()
            size = 0
        buf.append(e)
        size += len(e["text"])
    flush()
    # very long single paragraphs: split on sentence boundaries
    final = []
    for e in out:
        if e["type"] == "text" and len(e["text"]) > CHUNK * 1.6:
            sents = re.split(r"(?<=[.!?])\s+", e["text"])
            part = ""
            for s in sents:
                if len(part) + len(s) > CHUNK and part:
                    final.append({**e, "id": e["id"] + f"_{len(final)}", "text": part})
                    part = ""
                part += (" " if part else "") + s
            if part:
                final.append({**e, "id": e["id"] + f"_{len(final)}", "text": part})
        else:
            final.append(e)
    return final


def _link_mentions(elements):
    """For each figure/table, collect body sentences that reference it ('Figure 3 shows ...')."""
    labelled = {e["label"].lower(): e for e in elements if e["type"] in ("figure", "table") and e.get("label")}
    pat = re.compile(r"\b(fig(?:ure)?s?\.?|tables?)\s*(\d+[a-z]?)", re.I)
    for e in elements:
        if e["type"] != "text":
            continue
        for s in re.split(r"(?<=[.!?])\s+", e["text"]):
            for m in pat.finditer(s):
                kind = "table" if m.group(1).lower().startswith("tab") else "figure"
                tgt = labelled.get(f"{kind} {m.group(2)}".lower())
                if tgt and s not in tgt["mentions"] and len(tgt["mentions"]) < 6:
                    tgt["mentions"].append(s[:400])
