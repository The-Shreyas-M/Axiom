"""Vision module: captions for figures/tables + image Q&A via a vision model.

Captions turn visual content into retrievable text in the same embedding
space, so figure questions are answered by RAG. Figure-level chat still
sends the raw image to the vision model (true multimodal). Provider-agnostic
(Gemini or Ollama); see providers.py.
"""

from __future__ import annotations

import time

from providers import get_default_provider

MAX_FIGURES_CAPTIONED = 15  # guard against long captioning runs
CAPTION_PROMPT = (
    "You are analyzing a figure extracted from an academic research paper. "
    "Describe it in detail for someone who cannot see it: the type of visual "
    "(plot, chart, diagram, photograph), axes, labels, legend entries, data "
    "trends, and the key takeaway. Transcribe any readable text in the image. "
    "Keep it under 150 words."
)

ASK_IMAGE_PROMPT = (
    "This is a figure from a research paper. Answer the user's question about "
    "it based only on what is visible in the image. If the answer cannot be "
    "determined from the image, say so clearly."
)


class FigureCaptioner:
    """Generates captions for extracted figures and answers image questions."""

    def __init__(self, provider=None,
                 max_figures: int = MAX_FIGURES_CAPTIONED, sleep_sec: float = 0.2):
        self.provider = provider or get_default_provider()
        self.max_figures = max_figures
        self.sleep_sec = sleep_sec

    def caption_figure(self, image_path: str) -> str:
        text = self.provider.generate_vision(image_path, CAPTION_PROMPT)
        time.sleep(self.sleep_sec)  # give the GPU a breath between calls
        return text.strip()

    def caption_paper(self, figures) -> dict[int, str]:
        """Return {figure_index: caption} for up to max_figures figures."""
        captions: dict[int, str] = {}
        for figure in figures[: self.max_figures]:
            try:
                captions[figure.number] = self.caption_figure(figure.image_path)
            except Exception as exc:  # keep pipeline alive if one call fails
                captions[figure.number] = f"[Caption failed: {exc}]"
        return captions

    def ask_about_image(self, image_path: str, question: str) -> str:
        return self.provider.generate_vision(image_path, question, system=ASK_IMAGE_PROMPT).strip()
