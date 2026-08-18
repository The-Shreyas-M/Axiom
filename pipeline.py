"""Pipeline: PDF -> OCR all pages -> classify -> embed -> index.

Session persists to disk. Page refresh preserves everything.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

from classifier import PaperClassifier
from embeddings import Embedder, chunk_text
from llm import ResearchLLM
from ocr import ocr_pages
from pdf_parser import Paper, parse_pdf
from providers import get_text_provider, get_vision_provider
from retriever import Retriever, make_chunk_meta, make_figure_meta, make_table_meta
from vision import FigureCaptioner

CLASSIFY_PREFIX_CHARS = 4000
SESSION_DIR = os.path.join(os.path.dirname(__file__), ".axiom_session")
SESSION_FILE = os.path.join(SESSION_DIR, "state.pkl")


@dataclass
class ProcessedPaper:
    paper: Paper
    classification: list[tuple[str, float]]
    ocr_pages: list[int] = field(default_factory=list)


class SessionStore:
    def __init__(self):
        self.retriever = Retriever()
        self.papers: list[ProcessedPaper] = []
        self.paper_figures: list[tuple] = []
        self._processed_sources: dict[str, ProcessedPaper] = {}
        self._embedder: Optional[Embedder] = None
        self._classifier: Optional[PaperClassifier] = None
        self._llm: Optional[ResearchLLM] = None
        self._captioner: Optional[FigureCaptioner] = None
        self._text_provider = None
        self._vision_provider = None

    @property
    def text_provider(self):
        if self._text_provider is None:
            self._text_provider = get_text_provider()
        return self._text_provider

    @property
    def vision_provider(self):
        if self._vision_provider is None:
            self._vision_provider = get_vision_provider()
        return self._vision_provider

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def classifier(self) -> PaperClassifier:
        if self._classifier is None:
            self._classifier = PaperClassifier()
        return self._classifier

    @property
    def llm(self) -> ResearchLLM:
        if self._llm is None:
            self._llm = ResearchLLM(self.text_provider)
        return self._llm

    @property
    def captioner(self) -> FigureCaptioner:
        if self._captioner is None:
            self._captioner = FigureCaptioner(self.vision_provider)
        return self._captioner

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        os.makedirs(SESSION_DIR, exist_ok=True)
        with open(SESSION_FILE, "wb") as f:
            pickle.dump({
                "retriever_index": self.retriever.index,
                "retriever_meta": self.retriever.metadata,
                "papers": self.papers,
                "paper_figures": self.paper_figures,
                "processed_sources": self._processed_sources,
            }, f)

    def load(self) -> bool:
        if not os.path.exists(SESSION_FILE):
            return False
        try:
            with open(SESSION_FILE, "rb") as f:
                data = pickle.load(f)
            self.retriever.index = data["retriever_index"]
            self.retriever.metadata = data["retriever_meta"]
            self.papers = data["papers"]
            self.paper_figures = data["paper_figures"]
            self._processed_sources = data["processed_sources"]
            return True
        except Exception:
            return False

    # -- pipeline ----------------------------------------------------------
    def process_pdf(self, path: str, progress=None) -> ProcessedPaper:
        def tick(pct: float, msg: str):
            if progress is not None:
                progress(pct, desc=msg)

        key = os.path.normpath(os.path.abspath(path))
        if key in self._processed_sources:
            return self._processed_sources[key]

        tick(0.05, "Parsing PDF...")
        paper = parse_pdf(path)

        all_page_numbers = [p.number for p in paper.pages]
        tick(0.10, f"GPU OCR on {len(all_page_numbers)} pages...")
        ocr_results = ocr_pages(paper.path, all_page_numbers)
        for pno, text in ocr_results.items():
            if text.strip():
                paper.pages[pno - 1].text = text
        ocr_pages = list(ocr_results.keys())

        tick(0.55, "Classifying...")
        classification = self.classifier.predict_top3(paper.full_text[:CLASSIFY_PREFIX_CHARS])

        tick(0.70, "Embedding & indexing...")
        self._index_paper(paper)

        processed = ProcessedPaper(
            paper=paper, classification=classification, ocr_pages=ocr_pages,
        )
        self.papers.append(processed)
        for f in paper.figures:
            self.paper_figures.append(
                (f.image_path, f.caption or f.label or f"Fig. {f.number}",
                 f.label or f"Fig. {f.number}", paper.name)
            )
        self._processed_sources[key] = processed

        tick(0.95, "Saving session...")
        self.save()
        tick(1.0, "Done!")
        return processed

    def reset(self) -> None:
        self.retriever = Retriever()
        self.papers = []
        self.paper_figures = []
        self._processed_sources = {}
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)

    def _index_paper(self, paper: Paper) -> None:
        texts, meta = [], []
        for page in paper.pages:
            for i, chunk in enumerate(chunk_text(page.text)):
                texts.append(chunk)
                meta.append(make_chunk_meta(chunk, page.number, paper.name, i))
        for table in paper.tables:
            texts.append(table.text)
            meta.append(make_table_meta(table, paper.name))
        for figure in paper.figures:
            caption = figure.caption or figure.label or f"Figure {figure.number}"
            texts.append(caption)
            meta.append(make_figure_meta(figure, caption, paper.name))
        vectors = self.embedder.embed_texts(texts)
        self.retriever.add(vectors, meta)

    def caption_figure_on_demand(self, image_path: str) -> str:
        return self.captioner.caption_figure(image_path)

    def answer(self, question: str, k: int = 6) -> tuple[str, list]:
        hits = self.retriever.search(self.embedder.embed_query(question), k=k)
        answer = self.llm.answer(question, hits)
        figure_images = [h["image_path"] for h in hits if h.get("image_path")]
        return answer, figure_images

    def stats(self) -> str:
        lines = []
        for p in self.papers:
            paper = p.paper
            top = p.classification[0]
            top3 = ", ".join(f"{c} ({v:.0%})" for c, v in p.classification)
            lines.append(
                f"**{paper.name}**\n"
                f"- {top[0]} ({top[1]:.0%}) | Top 3: {top3}\n"
                f"- {paper.stats['pages']} pages · {len(self.retriever)} chunks · "
                f"{paper.stats['tables']} tables · {paper.stats['figures']} figures · "
                f"OCR: {len(p.ocr_pages)} pages"
            )
        return "\n\n".join(lines) if lines else "_No papers processed yet._"
