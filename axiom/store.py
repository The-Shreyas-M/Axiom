"""Per-paper vector store (Chroma, on disk) + BM25, hybrid retrieval with explicit Figure/Table resolution."""
import os
import re

import chromadb
from rank_bm25 import BM25Okapi

from . import config as C

_emb = None
_client = None
QPREFIX = "Represent this sentence for searching relevant passages: "


def embedder():
    global _emb
    if _emb is None:
        from fastembed import TextEmbedding
        _emb = TextEmbedding(C.EMBED_MODEL, cache_dir=str(C.DATA / "models"), threads=os.cpu_count())
    return _emb


def embed(texts, query=False):
    texts = [(QPREFIX + t) if query else t for t in texts]
    # length-sorted tiny batches: avoids padding every text to the longest one (3x faster on CPU)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = list(embedder().embed([texts[i] for i in order], batch_size=2))
    out = [None] * len(texts)
    for pos, i in enumerate(order):
        out[i] = vecs[pos].tolist()
    return out


def client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(C.DATA / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False))
    return _client


def index_text(e):
    """What gets embedded for an element: content + everything that gives it context."""
    head = f"[{e['section']}] " if e.get("section") else ""
    if e["type"] == "text":
        return head + e["text"]
    parts = [e.get("label") or e["type"].title(), e.get("caption", ""), e.get("desc", ""), e.get("text", "")[:1200]]
    parts += e.get("mentions", [])[:3]
    return head + " ".join(p for p in parts if p)


def build_index(pid, elements, progress=lambda p, m: None):
    name = f"p_{pid}"
    try:
        client().delete_collection(name)
    except Exception:
        pass
    col = client().create_collection(name, metadata={"hnsw:space": "cosine"})
    docs = [index_text(e) for e in elements]
    B = 128
    for i in range(0, len(docs), B):
        progress(i / max(1, len(docs)), f"Embedding {i}/{len(docs)}")
        chunk = elements[i:i + B]
        # truncate long docs: bge-small only reads 512 tokens anyway, so the rest is wasted compute
        col.add(ids=[e["id"] for e in chunk], documents=docs[i:i + B],
                embeddings=embed([d[:1500] for d in docs[i:i + B]]),
                metadatas=[{"type": e["type"], "page": e["page"]} for e in chunk])


def reindex_one(pid, e):
    col = client().get_collection(f"p_{pid}")
    d = index_text(e)
    col.upsert(ids=[e["id"]], documents=[d], embeddings=embed([d]), metadatas=[{"type": e["type"], "page": e["page"]}])


def delete_index(pid):
    try:
        client().delete_collection(f"p_{pid}")
    except Exception:
        pass


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class Retriever:
    def __init__(self, pid, elements):
        self.pid, self.els = pid, elements
        self.by_id = {e["id"]: e for e in elements}
        self.bm = BM25Okapi([_tok(index_text(e)) for e in elements]) if elements else None

    def search(self, q, k=6, focus=None):
        explicit = []
        for kind, num in re.findall(r"\b(fig(?:ure)?|table|tab)\.?\s*(\d+[a-z]?)", q, re.I):
            kind = "table" if kind.lower().startswith("tab") else "figure"
            want = f"{kind} {num}".lower()
            explicit += [e["id"] for e in self.els if e.get("label", "").lower() == want]
        pages = [int(p) for p in re.findall(r"\bpage\s*(\d+)", q, re.I)]
        scores = {}
        dense = client().get_collection(f"p_{self.pid}").query(
            query_embeddings=embed([q], query=True), n_results=min(12, len(self.els)))["ids"][0]
        sparse = []
        if self.bm:
            sc = self.bm.get_scores(_tok(q))
            sparse = [self.els[i]["id"] for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:12] if sc[i] > 0]
        for lst in (dense, sparse):
            for r, i in enumerate(lst):
                scores[i] = scores.get(i, 0) + 1 / (60 + r)
        for p in pages:
            for e in self.els:
                if e["page"] == p:
                    scores[e["id"]] = scores.get(e["id"], 0) + 0.02
        ranked = [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]
        out = []
        for i in ([focus] if focus else []) + explicit + ranked:
            if i in self.by_id and i not in out:
                out.append(i)
        return [self.by_id[i] for i in out[:k + (1 if focus else 0) + len(explicit)]]
