"""Vision module: on-demand figure captioning + image Q&A via provider vision API.

Figure captioning is NOT done during processing. Captions are generated
on-demand when a user clicks a figure in the gallery or asks about one.
This keeps initial processing fast (~5-10s instead of minutes).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

MAX_CAPTION_WORKERS = 3

CAPTION_PROMPT = (
    "Describe this research paper figure briefly: chart type, axes, labels, "
    "data trends, key takeaway. Under 80 words."
)

ASK_IMAGE_PROMPT = (
    "This is a figure from a research paper. Answer the question based only "
    "on what you see in the image. Be concise."
)


class FigureCaptioner:
    """On-demand figure captioning and image Q&A."""

    def __init__(self, provider=None):
        self.provider = provider

    def _get_provider(self):
        if self.provider is None:
            from providers import get_default_provider
            self.provider = get_default_provider()
        return self.provider

    def caption_figure(self, image_path: str) -> str:
        try:
            return self._get_provider().generate_vision(
                image_path, CAPTION_PROMPT
            ).strip()
        except Exception as exc:
            return f"[Caption error: {exc}]"

    def caption_figures_parallel(self, image_paths: list[str]) -> dict[int, str]:
        """Caption multiple figures in parallel. Returns {index: caption}."""
        if not image_paths:
            return {}
        if len(image_paths) == 1:
            return {0: self.caption_figure(image_paths[0])}

        captions: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=MAX_CAPTION_WORKERS) as pool:
            futures = {
                pool.submit(self.caption_figure, path): idx
                for idx, path in enumerate(image_paths)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    captions[idx] = future.result()
                except Exception:
                    captions[idx] = "[Caption unavailable]"
        return captions

    def ask_about_image(self, image_path: str, question: str) -> str:
        try:
            return self._get_provider().generate_vision(
                image_path, f"{ASK_IMAGE_PROMPT}\n\nQuestion: {question}"
            ).strip()
        except Exception as exc:
            return f"Error: {exc}"
