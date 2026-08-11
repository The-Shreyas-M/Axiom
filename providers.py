"""Provider factory: picks the LLM backend.

LLM_PROVIDER env overrides the choice:
  "gemini" -> GeminiProvider (uses GEMINI_API_KEY)
  "ollama" -> OllamaProvider (local, needs ollama serve)
  unset    -> gemini if GEMINI_API_KEY is set, else ollama
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # project root .env


def get_default_provider():
    choice = os.environ.get("LLM_PROVIDER", "")
    if not choice:
        choice = "gemini" if os.environ.get("GEMINI_API_KEY") else "ollama"

    if choice == "ollama":
        from ollama_llm import OllamaProvider

        return OllamaProvider()
    from gemini_provider import GeminiProvider

    return GeminiProvider()
