"""Retriever: FAISS index over all content with citation metadata.

Every vector maps to metadata: {type, page, label, text, image_path, paper}
so answers can cite page numbers and figure/table references.
"""

from __future__ import annotations

import pickle
from typing import Any

import faiss
import numpy as np

from embeddings import EMBED_DIM


class Retriever:
    def __init__(self, dim: int = EMBED_DIM):
        self.index = faiss.IndexFlatIP(dim)  # cosine sim on normalized vectors
        self.metadata: list[dict[str, Any]] = []

    def add(self, vectors: np.ndarray, meta_list: list[dict[str, Any]]) -> None:
        assert vectors.shape[0] == len(meta_list)
        self.index.add(vectors)
        self.metadata.extend(meta_list)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(query_vec.reshape(1, -1), k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            hit = dict(self.metadata[int(idx)])
            hit["score"] = float(score)
            hits.append(hit)
        return hits

    def __len__(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.index.d)
        self.metadata = []

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"index": faiss.serialize_index(self.index), "metadata": self.metadata}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = faiss.deserialize_index(data["index"])
        self.metadata = data["metadata"]


def make_chunk_meta(chunk: str, page: int, paper: str, chunk_idx: int) -> dict:
    return {"type": "text", "page": page, "label": f"p.{page}", "text": chunk,
            "image_path": None, "paper": paper, "chunk_idx": chunk_idx}


def make_table_meta(table, paper: str) -> dict:
    return {"type": "table", "page": table.page, "label": f"Table {table.number}",
            "text": table.text, "image_path": None, "paper": paper}


def make_figure_meta(figure, caption: str, paper: str) -> dict:
    label = figure.label or f"Fig. {figure.number}"
    return {"type": "figure", "page": figure.page, "label": label,
            "text": caption, "image_path": figure.image_path, "paper": paper}
