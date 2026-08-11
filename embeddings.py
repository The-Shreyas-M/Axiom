"""Embeddings: sentence chunking + SBERT (BAAI/bge-small-en-v1.5).

All content (text chunks, tables, figure captions) lives in one vector space,
so retrieval can mix paragraphs, table cells and figure descriptions.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
CHUNK_SIZE = 600  # chars
CHUNK_OVERLAP = 100

# bge models recommend a query-side instruction for retrieval
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks on paragraph boundaries."""
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # back off to the nearest paragraph/sentence boundary
            cut = text.rfind("\n", start, end)
            if cut == -1 or cut < start + chunk_size // 2:
                cut = text.rfind(". ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 40]


class Embedder:
    """Thin wrapper around a SentenceTransformer with normalized vectors."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype(np.float32).reshape(1, -1)
