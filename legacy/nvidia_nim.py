"""NVIDIA NIM provider: cloud text + vision generation, OpenAI-compatible API.

Uses https://integrate.api.nvidia.com/v1 (NVIDIA API / build.nvidia.com).
Key: NVIDIA_API_KEY (alias NVAPI_KEY) env var or .env in the project root.
Get a free key at https://build.nvidia.com

Env overrides:
  NVIDIA_MODEL         default text model (fallback chain tried in order)
  NVIDIA_VISION_MODEL  default vision model (fallback chain tried in order)
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

BASE_URL = "https://integrate.api.nvidia.com/v1"
TIMEOUT = 300

DEFAULT_TEXT_MODELS = [
    os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nvidia-nemotron-nano-9b-v2",
]
DEFAULT_VISION_MODELS = [
    os.environ.get("NVIDIA_VISION_MODEL", "meta/llama-3.2-11b-vision-instruct"),
    "nvidia/nemotron-nano-12b-v2-vl",
]


def resolve_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVAPI_KEY", "")
    if not key:
        raise RuntimeError(
            "No NVIDIA_API_KEY found. Set it via environment variable or a .env "
            "file in the project root (see https://build.nvidia.com)."
        )
    return key


class NvidiaNimProvider:
    max_context_chars = 400_000

    def __init__(self, api_key: Optional[str] = None,
                 text_models: Optional[list[str]] = None,
                 vision_models: Optional[list[str]] = None,
                 base_url: str = BASE_URL):
        self.api_key = api_key or resolve_api_key()
        self.text_models = text_models or list(DEFAULT_TEXT_MODELS)
        self.vision_models = vision_models or list(DEFAULT_VISION_MODELS)
        self.base_url = base_url.rstrip("/")

    # -- internals ---------------------------------------------------------
    def _chat(self, messages: list[dict], models: list[str]) -> str:
        """Try each model in the fallback chain until one responds."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_exc: Exception | None = None
        for model in models:
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 2048,
                        "temperature": 0.3,
                        "stream": False,
                    },
                    headers=headers,
                    timeout=TIMEOUT,
                )
                if resp.status_code == 404:
                    raise RuntimeError(f"model '{model}' not found on NVIDIA API")
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if content:
                    return content.strip()
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"NVIDIA API call failed: {last_exc}")

    # -- interface (mirrors GeminiProvider / OllamaProvider) ---------------
    def generate_text(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat(messages, self.text_models)

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
        return self._chat(messages, self.vision_models)

    def check(self) -> tuple[bool, str]:
        try:
            text = self._chat(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                self.text_models,
            )
            return True, f"NVIDIA ready — model '{self.text_models[0]}' (responded: {text[:10]})"
        except Exception as exc:
            return False, str(exc)
