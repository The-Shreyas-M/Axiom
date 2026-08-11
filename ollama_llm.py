"""Ollama provider: local text + vision generation. No API keys.

Requires: `ollama serve` running (default http://127.0.0.1:11434) and a
multimodal model pulled, e.g. `ollama pull gemma3:4b` (6GB-class GPUs) or
`ollama pull qwen3-vl:8b` (Colab T4).

Env overrides:
  LLM_MODEL        default model name (default "gemma3:4b")
  OLLAMA_BASE_URL  server URL (default "http://127.0.0.1:11434")
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import httpx

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
TIMEOUT = 900  # seconds; local models are slow on big prompts


class OllamaProvider:
    """Text and vision generation through Ollama's /api/chat endpoint."""

    max_context_chars = 30_000

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    # -- helpers -----------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        try:
            resp = httpx.post(f"{self.base_url}{path}", json=payload, timeout=TIMEOUT)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Ollama server not reachable. Start it first:\n"
                "  ollama serve   (or open the Ollama app)"
            ) from exc
        if resp.status_code == 404 and "model" in resp.text.lower():
            raise RuntimeError(
                f"Model '{self.model}' is not pulled. Run:\n"
                f"  ollama pull {self.model}"
            )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        try:
            resp = httpx.get(f"{self.base_url}{path}", timeout=TIMEOUT)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                "Ollama server not reachable. Start it first:\n"
                "  ollama serve   (or open the Ollama app)"
            ) from exc
        resp.raise_for_status()
        return resp.json()

    def _image_b64(self, image_path: str) -> str:
        return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")

    # -- API ---------------------------------------------------------------
    def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = self._post("/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3},
        })
        return (data.get("message") or {}).get("content", "").strip()

    def generate_vision(self, image_path: str, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [self._image_b64(image_path)],
        })
        data = self._post("/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3},
        })
        return (data.get("message") or {}).get("content", "").strip()

    def check(self) -> tuple[bool, str]:
        """Return (ok, message). Verifies server is up and model is pulled."""
        try:
            tags = self._get("/api/tags")
        except RuntimeError as exc:
            return False, str(exc)
        names = [t.get("name") for t in tags.get("models", [])]
        base = self.model.split(":")[0]
        if not any(n == self.model or n.startswith(base + ":") for n in names):
            return False, f"Model '{self.model}' not pulled. Run: ollama pull {self.model}"
        return True, f"Ollama ready — model '{self.model}'"
