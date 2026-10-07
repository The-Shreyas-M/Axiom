"""Gemini provider: cloud text + vision generation via the Interactions API."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash-lite"]


def resolve_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found.")
    return key


class GeminiProvider:
    max_context_chars = 500_000

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from google import genai
        self.client = genai.Client(api_key=api_key or resolve_api_key())
        self.model = model or PRIMARY_MODEL
        self._fallbacks = [m for m in FALLBACK_MODELS if m != self.model]

    def _image_part(self, image_path: str) -> dict:
        mime, _ = mimetypes.guess_type(image_path)
        return {
            "type": "image",
            "mime_type": mime or "image/png",
            "data": base64.b64encode(Path(image_path).read_bytes()).decode("ascii"),
        }

    def _call(self, input_) -> str:
        models = [self.model] + self._fallbacks
        last_exc: Exception | None = None
        for model in models:
            try:
                resp = self.client.interactions.create(model=model, input=input_)
                text = getattr(resp, "output_text", None) or ""
                if not text:
                    for out in getattr(resp, "outputs", []) or []:
                        if getattr(out, "text", None):
                            text = out.text
                            break
                return text.strip()
            except Exception as exc:
                last_exc = exc
                if "denied access" in str(exc).lower():
                    break
                time.sleep(0.3)
        raise RuntimeError(f"Gemini call failed: {last_exc}")

    def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        user = f"{system}\n\n{prompt}" if system else prompt
        return self._call(user)

    def generate_vision(self, image_path: str, prompt: str, system: Optional[str] = None) -> str:
        user = f"{system}\n\n{prompt}" if system else prompt
        return self._call([self._image_part(image_path), {"type": "text", "text": user}])

    def check(self) -> tuple[bool, str]:
        try:
            text = self._call("Reply with exactly: OK")
            return True, f"Gemini ready — model '{self.model}' (responded: {text[:10]})"
        except Exception as exc:
            return False, str(exc)
