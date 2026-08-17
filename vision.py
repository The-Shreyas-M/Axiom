"""Vision module: captions for figures/tables + image Q&A via the provider's vision API.

Uses the provider's native vision capability (Gemini, NVIDIA NIM, Groq, etc.)
with parallel figure captioning via ThreadPoolExecutor for speed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

MAX_FIGURES_CAPTIONED = 5
MAX_WORKERS = 5

CAPTION_PROMPT = (
    "Describe this research paper figure in detail: the type of visual "
    "(plot, chart, diagram, photograph), axes, labels, legend entries, data "
    "trends, and the key takeaway. Keep it under 100 words."
)

ASK_IMAGE_PROMPT = (
    "This is a figure from a research paper. Answer the user's question about "
    "it based only on what is visible in the image. If the answer cannot be "
    "determined from the image, say so clearly."
)


class FigureCaptioner:
    """Generates captions for extracted figures and answers image questions."""

    def __init__(self, provider=None, max_figures: int = MAX_FIGURES_CAPTIONED,
                 max_workers: int = MAX_WORKERS):
        self.provider = provider
        self.max_figures = max_figures
        self.max_workers = max_workers

    def _get_provider(self):
        if self.provider is None:
            from providers import get_default_provider
            self.provider = get_default_provider()
        return self.provider

    def caption_figure(self, image_path: str) -> str:
        provider = self._get_provider()
        try:
            return provider.generate_vision(image_path, CAPTION_PROMPT).strip()
        except Exception:
            return "[Caption unavailable]"

    def caption_paper(self, figures) -> dict[int, str]:
        captions: dict[int, str] = {}
        to_caption = [(f.number, f.image_path) for f in figures[:self.max_figures]]

        if len(to_caption) <= 1:
            for num, path in to_caption:
                try:
                    captions[num] = self.caption_figure(path)
                except Exception:
                    captions[num] = "[Caption unavailable]"
            return captions

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.caption_figure, path): num
                for num, path in to_caption
            }
            for future in as_completed(futures):
                num = futures[future]
                try:
                    captions[num] = future.result()
                except Exception:
                    captions[num] = "[Caption unavailable]"

        return captions

    def ask_about_image(self, image_path: str, question: str) -> str:
        provider = self._get_provider()
        try:
            return provider.generate_vision(image_path, ASK_IMAGE_PROMPT + f"\n\nQuestion: {question}").strip()
        except Exception as exc:
            return f"Unable to answer: {exc}"
