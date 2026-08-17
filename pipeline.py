"""Pipeline: PDF(s) -> parse -> classify -> caption -> embed -> index."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from classifier import PaperClassifier
from embeddings import Embedder, chunk_text
from layout import LayoutAnalyzer, apply_layout
from llm import ResearchLLM
from ocr import OCR
from pdf_parser import Paper, parse_pdf
from providers import get_default_provider
from retriever import Retriever, make_chunk_meta, make_figure_meta, make_table_meta
from vision import FigureCaptioner

CLASSIFY_PREFIX_CHARS = 4000


@dataclass
class ProcessedPaper:
    paper: Paper
    classification: list[tuple[str, float]]
    captions: dict[int, str]
    ocr_text_pages: list[int] = field(default_factory=list)


class SessionStore:
    """Shared in-memory state: one FAISS index + lazy-loaded models."""

    def __init__(self):
        self.retriever = Retriever()
        self.papers: list[ProcessedPaper] = []
        self.paper_figures: list[tuple] = []
        self._processed_sources: dict[str, ProcessedPaper] = {}
        self._embedder: Optional[Embedder] = None
        self._classifier: Optional[PaperClassifier] = None
        self._llm: Optional[ResearchLLM] = None
        self._captioner: Optional[FigureCaptioner] = None
        self._provider = get_default_provider()
        self._ocr = OCR()
        self._layout = LayoutAnalyzer()

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
            self._llm = ResearchLLM(self._provider)
        return self._llm

    @property
    def captioner(self) -> FigureCaptioner:
        if self._captioner is None:
            self._captioner = FigureCaptioner(self._provider)
        return self._captioner

    def process_pdf(self, path: str, use_ocr: bool = False, progress=None) -> ProcessedPaper:
        def tick(msg: str):
            if progress is not None:
                progress(0.0, desc=msg)

        key = os.path.normpath(os.path.abspath(path))
        if key in self._processed_sources:
            tick("Already processed")
            return self._processed_sources[key]

        tick("Parsing PDF...")
        paper = parse_pdf(path)

        ocr_text_pages: list[int] = []
        if use_ocr and paper.scanned_pages:
            if not self._ocr.available:
                raise RuntimeError("OCR requested but no engine installed.")
            if self._layout.available:
                tick(f"OCR + layout on {len(paper.scanned_pages)} page(s)...")
                try:
                    layouts = self._layout.analyze(paper.path, paper.scanned_pages)
                    ocr_text_pages = apply_layout(paper, layouts)
                    missing = [pno for pno in paper.scanned_pages if pno not in ocr_text_pages]
                    if missing:
                        for pno, text in self._ocr.ocr_pages(paper.path, missing).items():
                            if text:
                                ocr_text_pages.append(pno)
                                paper.pages[pno - 1].text = text
                except Exception:
                    for pno, text in self._ocr.ocr_pages(paper.path, paper.scanned_pages).items():
                        if text:
                            ocr_text_pages.append(pno)
                            paper.pages[pno - 1].text = text
            else:
                for pno, text in self._ocr.ocr_pages(paper.path, paper.scanned_pages).items():
                    if text:
                        ocr_text_pages.append(pno)
                        paper.pages[pno - 1].text = text

        tick("Classifying & captioning figures...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            cls_future = pool.submit(
                self.classifier.predict_top3, paper.full_text[:CLASSIFY_PREFIX_CHARS]
            )
            cap_future = None
            if paper.figures:
                cap_future = pool.submit(self.captioner.caption_paper, paper.figures)

            classification = cls_future.result()
            captions = cap_future.result() if cap_future else {}

        tick("Embedding & indexing...")
        self._index_paper(paper, captions)

        processed = ProcessedPaper(
            paper=paper, classification=classification,
            captions=captions, ocr_text_pages=ocr_text_pages,
        )
        self.papers.append(processed)
        for f in paper.figures:
            self.paper_figures.append(
                (f.image_path, captions.get(f.number, f.caption or f.label),
                 f.label or f"Fig. {f.number}", paper.name)
            )
        self._processed_sources[key] = processed
        return processed

    def is_processed(self, path: str) -> bool:
        return os.path.normpath(os.path.abspath(path)) in self._processed_sources

    def reset(self) -> None:
        self.retriever = Retriever()
        self.papers = []
        self.paper_figures = []
        self._processed_sources = {}

    def _index_paper(self, paper: Paper, captions: dict[int, str]) -> None:
        texts: list[str] = []
        meta: list[dict] = []
        for page in paper.pages:
            for i, chunk in enumerate(chunk_text(page.text)):
                texts.append(chunk)
                meta.append(make_chunk_meta(chunk, page.number, paper.name, i))
        for table in paper.tables:
            texts.append(table.text)
            meta.append(make_table_meta(table, paper.name))
        for figure in paper.figures:
            caption = captions.get(figure.number) or figure.caption or figure.label
            texts.append(caption)
            meta.append(make_figure_meta(figure, caption, paper.name))

        vectors = self.embedder.embed_texts(texts)
        self.retriever.add(vectors, meta)

    def answer(self, question: str, k: int = 6) -> tuple[str, list]:
        hits = self.retriever.search(self.embedder.embed_query(question), k=k)
        answer = self.llm.answer(question, hits)
        figure_images = [h["image_path"] for h in hits if h.get("image_path")]
        return answer, figure_images

    def stats(self) -> str:
        lines = []
        for processed in self.papers:
            paper = processed.paper
            top = processed.classification[0]
            top3 = ", ".join(f"{c} ({p:.0%})" for c, p in processed.classification)
            lines.append(
                f"**{paper.name}**  \n"
                f"- Category: **{top[0]}** ({top[1]:.0%}) | Top 3: {top3}  \n"
                f"- {paper.stats['pages']} pages · {len(self.retriever)} chunks · "
                f"{paper.stats['tables']} tables · {paper.stats['figures']} figures  \n"
                f"- OCR: {len(processed.ocr_text_pages)} pages · "
                f"Captioned: {sum(1 for c in processed.captions.values() if c and not c.startswith('[Caption'))}"
            )
        return "\n\n".join(lines) if lines else "_No papers processed yet._"
