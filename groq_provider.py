"""Groq provider: fast LLMs via Groq's OpenAI‑compatible API.

Endpoint: https://api.groq.com/openai/v1/chat/completions
Key: GROQ_API_KEY env var or .env in the project root.
Get a free key at https://console.groq.com/

Env overrides:
  GROQ_MODEL   default model name (default "mixtral-8x7b-32768")
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import httpx

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "mixtral-8x7b-32768")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT = 30.0  # seconds


def resolve_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "No GROQ_API_KEY found. Set it via environment variable or a .env "
            "file in the project root (see https://console.groq.com/)."
        )
    return key


class GroqProvider:
    """Text and vision generation through Groq's chat endpoint."""

    max_context_chars = 32_000  # conservative; actual limit depends on model

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or resolve_api_key()
        self.model = model or DEFAULT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # -- internals ---------------------------------------------------------
    def _chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": False,
        }
        try:
            resp = httpx.post(API_URL, json=payload, headers=self.headers, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content:
                return content.strip()
            # Some Groq responses may put text in a different field
            for choice in data.get("choices", []):
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            raise RuntimeError("Empty response from Groq API")
        except httpx.HTTPStatusError as exc:
            # Include Groq's error message if possible
            detail = exc.response.text
            raise RuntimeError(f"Groq API error {exc.response.status_code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

    # -- interface (mirrors GeminiProvider / OllamaProvider) ---------------
    def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat(messages)

    def generate_vision(self, image_path: str, prompt: str, system: Optional[str] = None) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        suffix = image_path.lower()
        mime = "image/jpeg" if suffix.endswith((".jpg", ".jpeg")) else "image/png"
        text = f"{system}\n\n{prompt}" if system else prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
        return self._chat(messages)

    def check(self) -> tuple[bool, str]:
        try:
            text = self._chat([{"role": "user", "content": "Reply with exactly: OK"}])
            return True, f"Groq ready — model '{self.model}' (responded: {text[:10]})"
        except Exception as exc:
            return False, str(exc)