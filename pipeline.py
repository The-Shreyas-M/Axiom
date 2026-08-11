"""Pipeline: PDF(s) -> parse -> classify -> caption figures -> embed -> index.

All state (retriever, models, figures) lives in a module-level SessionStore so
the Gradio app can process papers, then answer questions against the index.
"""

from __future__ import annotations

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
        self.paper_figures: list[tuple] = []  # (image_path, caption, label, paper_name)
        self._embedder: Optional[Embedder] = None
        self._classifier: Optional[PaperClassifier] = None
        self._llm: Optional[ResearchLLM] = None
        self._captioner: Optional[FigureCaptioner] = None
        self._provider = get_default_provider()
        self._ocr = OCR()
        self._layout = LayoutAnalyzer()

    # -- lazy models -------------------------------------------------------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def classifier(self) -> PaperClassifier:
        if self._classifier is None:
            self._classifier = PaperClassifier()
            self._classifier.train()
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

    # -- pipeline ----------------------------------------------------------
    def process_pdf(self, path: str, use_ocr: bool = False, caption_figures: bool = True,
                    progress=None) -> ProcessedPaper:
        def tick(msg: str):
            if progress is not None:
                progress(0.0, desc=msg)

        tick("Parsing PDF (text, tables, figures)...")
        paper = parse_pdf(path)

        ocr_text_pages: list[int] = []
        if use_ocr and paper.scanned_pages:
            if not self._ocr.available:
                raise RuntimeError(
                    "OCR requested but no OCR engine is installed. Install one with "
                    "`pip install -r requirements-ocr.txt` (paddleocr is recommended, "
                    "easyocr as fallback)."
                )
            if self._layout.available:
                tick(f"OCR + layout analysis on {len(paper.scanned_pages)} scanned page(s)...")
                try:
                    layouts = self._layout.analyze(paper.path, paper.scanned_pages)
                    ocr_text_pages = apply_layout(paper, layouts)
                    missing = [pno for pno in paper.scanned_pages if pno not in ocr_text_pages]
                    if missing:
                        tick(f"OCR fallback for {len(missing)} page(s)...")
                        for pno, text in self._ocr.ocr_pages(paper.path, missing).items():
                            if text:
                                ocr_text_pages.append(pno)
                                paper.pages[pno - 1].text = text
                except Exception as exc:
                    tick(f"Layout analysis failed ({exc}); using OCR text only")
                    for pno, text in self._ocr.ocr_pages(paper.path, paper.scanned_pages).items():
                        if text:
                            ocr_text_pages.append(pno)
                            paper.pages[pno - 1].text = text
            else:
                tick(f"OCR on {len(paper.scanned_pages)} scanned page(s)...")
                for pno, text in self._ocr.ocr_pages(paper.path, paper.scanned_pages).items():
                    if text:
                        ocr_text_pages.append(pno)
                        paper.pages[pno - 1].text = text

        tick("Classifying paper (stemming + lemmatization)...")
        classification = self.classifier.predict_top3(paper.full_text[:CLASSIFY_PREFIX_CHARS])

        captions: dict[int, str] = {}
        if caption_figures and paper.figures:
            try:
                tick(f"Captioning up to {len(paper.figures)} figures with vision model...")
                captions = self.captioner.caption_paper(paper.figures)
            except Exception as exc:
                captions = {f.number: f.caption or f.label for f in paper.figures}
                tick(f"Figure captioning skipped: {exc}")

        tick("Chunking + embedding + indexing (FAISS)...")
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
        return processed

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

    # -- queries -----------------------------------------------------------
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
                f"- Predicted category: **{top[0]}** ({top[1]:.0%} confidence)  \n"
                f"- Top 3: {top3}  \n"
                f"- {paper.stats['pages']} pages · {len(self.retriever)} indexed chunks · "
                f"{paper.stats['tables']} tables · {paper.stats['figures']} figures  \n"
                f"- Scanned pages: {paper.stats['scanned_pages']} · "
                f"OCR applied: {len(processed.ocr_text_pages)}  \n"
                f"- Figures captioned: {sum(1 for c in processed.captions.values() if c and not c.startswith('[Caption'))}"
            )
        return "\n\n".join(lines) if lines else "_No papers processed yet._"
