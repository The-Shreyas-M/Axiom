"""Ingest pipeline + background VLM description of figures/tables."""
import hashlib, json, shutil, threading, time

from . import config as C, llm, store
from .parser import parse_pdf

DESC_PROMPT = ("You are analysing a figure/table from a research paper.{cap} Describe precisely what it shows: "
               "type of chart/diagram, axes, labels, legend entries, key numbers, trends and the main takeaway. "
               "If it is a table, transcribe the headers and the most important rows. Be concise (under 150 words).")


def paper_dir(pid):
    return C.DATA / "papers" / pid


def load(pid):
    d = paper_dir(pid)
    return (json.loads((d / "meta.json").read_text(encoding="utf8")),
            json.loads((d / "elements.json").read_text(encoding="utf8")))


def save_elements(pid, els):
    (paper_dir(pid) / "elements.json").write_text(json.dumps(els, ensure_ascii=False), encoding="utf8")


def save_meta(pid, meta):
    (paper_dir(pid) / "meta.json").write_text(json.dumps(meta), encoding="utf8")


def list_papers():
    out = []
    for m in sorted((C.DATA / "papers").glob("*/meta.json"), key=lambda p: -p.stat().st_mtime):
        out.append(json.loads(m.read_text(encoding="utf8")))
    return out


def guess_title(pdf):
    """PDF metadata title if sensible, else the largest-font line in the top half of page 1."""
    import fitz
    try:
        doc = fitz.open(pdf)
        t = (doc.metadata or {}).get("title", "").strip()
        if 8 <= len(t) <= 200 and not t.lower().endswith((".pdf", ".doc", ".docx", ".tex")):
            return t
        best, size = "", 0
        for b in doc[0].get_text("dict")["blocks"]:
            if b["type"] != 0 or b["bbox"][1] > doc[0].rect.height * 0.5:
                continue
            for ln in b["lines"]:
                txt = "".join(sp["text"] for sp in ln["spans"]).strip()
                sz = max((sp["size"] for sp in ln["spans"]), default=0)
                if len(txt) > 6 and sz > size + 0.5:
                    best, size = txt, sz
        return best[:150]
    except Exception:
        return ""


def ingest(pdf_bytes, filename, progress):
    pid = hashlib.sha1(pdf_bytes).hexdigest()[:12]
    d = paper_dir(pid)
    if (d / "meta.json").exists():
        progress(1.0, "Already ingested")
        return pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "paper.pdf").write_bytes(pdf_bytes)
    t0 = time.time()
    els, stats = parse_pdf(d / "paper.pdf", d, lambda p, m: progress(p * 0.7, m))
    save_elements(pid, els)
    store.build_index(pid, els, lambda p, m: progress(0.7 + p * 0.3, m))
    title = guess_title(d / "paper.pdf") or next((e["text"] for e in els if e["type"] == "text"), filename)[:120]
    meta = {"id": pid, "filename": filename, "title": title, "stats": stats, "seconds": round(time.time() - t0, 1),
            "described": 0, "to_describe": sum(1 for e in els if e["type"] == "figure" and e["image"]),
            "created": time.time()}
    save_meta(pid, meta)
    progress(1.0, "Done")
    def after():
        from . import qa
        qa.start_summary(pid)  # summary first, so it is streaming/ready when the user opens the paper
        while pid in qa._sum:
            time.sleep(0.5)
        if C.DESCRIBE_IN_BACKGROUND:
            describe_all(pid)
    threading.Thread(target=after, daemon=True).start()
    return pid


_desc_lock = threading.Lock()
_inter = 0
_inter_lock = threading.Lock()


def interactive_start():
    global _inter
    with _inter_lock:
        _inter += 1


def interactive_end():
    global _inter
    with _inter_lock:
        _inter -= 1


def describe_element(pid, e, yield_to_user=False):
    """Returns True ok, False if LLM unavailable, None if preempted by a user request."""
    cap = f" Its caption: \"{e['caption']}\"." if e.get("caption") else ""
    msg = [{"role": "user", "content": DESC_PROMPT.format(cap=cap)}]
    toks = []
    for t in llm.chat_stream(msg, model=C.VLM, images=[paper_dir(pid) / e["image"]], max_tokens=200):
        toks.append(t)
        if yield_to_user and _inter > 0:
            return None  # user is asking something: abort now so they get the GPU
    d = "".join(toks).strip()
    if "[Cannot reach Ollama" in d or "[LLM error" in d:
        return False
    e["desc"] = d
    return True


def describe_all(pid):
    """Background, lowest priority: describe image-only figures/tables so retrieval 'sees' them.
    Digital tables already carry their text, so they are skipped."""
    _, els = load(pid)
    todo = [e["id"] for e in els if e["image"] and not e["desc"] and
            (e["type"] == "figure" or (e["type"] == "table" and len(e.get("text", "")) < 80))]
    for eid in todo:
        while _inter > 0:
            time.sleep(0.3)
        with _desc_lock:
            meta, els = load(pid)  # fresh copy: a manual "Analyse image" may have saved meanwhile
            e = next((x for x in els if x["id"] == eid), None)
            if e is None or e["desc"]:
                continue
            r = describe_element(pid, e, yield_to_user=True)
            if r is False:
                return  # LLM unavailable
            if r is None:
                continue  # preempted; stays undescribed until the user asks
            save_elements(pid, els)
            store.reindex_one(pid, e)
            meta["described"] = sum(1 for x in els if x["desc"])
            save_meta(pid, meta)


def delete(pid):
    store.delete_index(pid)
    shutil.rmtree(paper_dir(pid), ignore_errors=True)
