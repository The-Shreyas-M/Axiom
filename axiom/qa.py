"""Grounded Q&A and summary over a paper's elements."""
import threading
import time

from . import config as C, ingest, llm, store

SYS = ("You are Axiom, a careful research-paper assistant. Answer ONLY from the provided paper excerpts "
       "(and the attached image, if any). Cite sources inline like [p3] or [Figure 2] or [Table 1]. "
       "If the excerpts do not contain the answer, say so plainly. Be concise and precise; keep numbers exact.")
BUDGET = 3800       # chars of context for text-only questions
BUDGET_IMG = 2000   # smaller when an image is attached (image tokens already cost prefill time)


def retriever(pid):
    return store.Retriever(pid, ingest.load(pid)[1])  # cheap; always sees fresh VLM descriptions


def invalidate(pid):
    pass


def tag(e):
    if e["type"] == "caption":
        return f"{e.get('label', 'Caption')} caption"
    return e.get("label") or f"{e['type']} p{e['page']}"


def render(e, full=False):
    cap = 1800 if full else 900
    head = f"[{tag(e)} | page {e['page']} | {e.get('section', '')}]"
    if e["type"] == "text":
        return f"{head}\n{e['text'][:cap * 2]}"
    parts = []
    if e.get("caption"):
        parts.append("Caption: " + e["caption"])
    if e.get("desc"):
        parts.append("Visual description: " + e["desc"])
    if e.get("text") and e["type"] != "caption":
        parts.append(("Table content:\n" if e["type"] == "table" else "Text inside figure: ") + e["text"][:cap])
    if e["type"] == "caption":
        parts.append(e["text"])
    if e.get("mentions"):
        parts.append("Referenced in text: " + " ".join(e["mentions"][:3]))
    return head + "\n" + "\n".join(parts)


def answer_stream(pid, question, focus=None, history=()):
    """Yields dicts: {'sources': [...]} first, then {'token': str}."""
    r = retriever(pid)
    hits = r.search(question, k=5, focus=focus)
    has_img = bool(focus and r.by_id.get(focus, {}).get("image")) or \
        any(e.get("image") and e["type"] in ("figure", "table") for e in hits[:2])
    budget = BUDGET_IMG if has_img else BUDGET
    ctx, used, total = [], [], 0
    for e in hits:
        s = render(e, full=(e["id"] == focus))
        if total + len(s) > budget and used:
            continue
        ctx.append(s)
        used.append(e)
        total += len(s)
    # attach image: the focused element, else a figure/table explicitly asked for
    img = None
    pdir = ingest.paper_dir(pid)
    pick = next((e for e in used if e["id"] == focus and e.get("image")), None) or \
        next((e for e in used[:2] if e.get("image") and e["type"] in ("figure", "table")), None)
    if pick:
        img = [pdir / pick["image"]]
    yield {"sources": [{"id": e["id"], "type": e["type"], "page": e["page"], "label": tag(e),
                        "image": e.get("image")} for e in used]}
    msgs = [{"role": "system", "content": SYS}]
    for h in list(history)[-4:]:
        msgs.append({"role": h["role"], "content": h["content"][:1200]})
    focus_note = ""
    if focus and focus in r.by_id:
        focus_note = f"The user is asking specifically about {tag(r.by_id[focus])} (listed first).\n\n"
    msgs.append({"role": "user", "content": "Paper excerpts:\n\n" + "\n\n---\n".join(ctx) +
                 f"\n\n---\n{focus_note}Question: {question}"})
    ingest.interactive_start()  # pauses background figure description so this answer gets the GPU
    try:
        for tok in llm.chat_stream(msgs, images=img, model=C.VLM if img else C.MODEL,
                                   max_tokens=450 if img else 600):
            yield {"token": tok}
    finally:
        ingest.interactive_end()


# ---------------------------------------------------------------- summary (generated once, shared, streamed)
_sum = {}
_sum_lock = threading.Lock()


def _summary_prompt(pid):
    meta, els = ingest.load(pid)
    text = [e for e in els if e["type"] == "text"]

    def sec(*names):
        return " ".join(e["text"] for e in text if any(n in e["section"].lower() for n in names))

    heads = []
    for e in text:
        if e["section"] not in heads:
            heads.append(e["section"])
    parts = {
        "Abstract": sec("abstract") or " ".join(e["text"] for e in text[:2]),
        "Introduction": sec("introduction")[:1000],
        "Results": sec("result", "experiment", "evaluation")[:1000],
        "Conclusion": sec("conclusion", "discussion")[:1000],
    }
    figs = [f"{e['label']}: {e['caption'][:120]}" for e in els
            if e["type"] in ("figure", "table") and e.get("caption")][:6]
    ctx = "\n\n".join(f"## {k}\n{v[:1400]}" for k, v in parts.items() if v)
    ctx += "\n\n## Sections\n" + "; ".join(heads[:20]) + "\n\n## Figures & tables\n" + "\n".join(figs)
    return ("Summarise this research paper for a busy researcher. Use exactly these markdown sections: "
            "**TL;DR** (2 sentences), **Problem**, **Approach**, **Key results** (bullets with exact numbers), "
            "**Limitations**. Be brief. Use only the text below.\n\n" + ctx)


def start_summary(pid):
    """Idempotent: begin generating in a thread; any number of viewers tail the same buffer."""
    with _sum_lock:
        if pid in _sum or (ingest.paper_dir(pid) / "summary.md").exists():
            return
        job = _sum[pid] = {"buf": [], "done": False}

    def run():
        try:
            for tok in llm.chat_stream([{"role": "user", "content": _summary_prompt(pid)}], max_tokens=500):
                job["buf"].append(tok)
        finally:
            out = "".join(job["buf"])
            if out and "[Cannot reach Ollama" not in out and "[LLM error" not in out:
                (ingest.paper_dir(pid) / "summary.md").write_text(out, encoding="utf8")
            job["done"] = True
            with _sum_lock:
                _sum.pop(pid, None)

    threading.Thread(target=run, daemon=True).start()


def summary_stream(pid):
    cache = ingest.paper_dir(pid) / "summary.md"
    if cache.exists():
        yield {"token": cache.read_text(encoding="utf8")}
        return
    start_summary(pid)
    job = _sum.get(pid)
    if job is None:  # finished already
        if cache.exists():
            yield {"token": cache.read_text(encoding="utf8")}
        return
    i = 0
    while True:
        buf = job["buf"]
        while i < len(buf):
            yield {"token": buf[i]}
            i += 1
        if job["done"] and i >= len(buf):
            return
        time.sleep(0.05)
