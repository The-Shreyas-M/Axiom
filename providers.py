"""Provider factory: picks the LLM backend based on env vars.

Priority when LLM_PROVIDER is unset:
  1. groq (fastest, if GROQ_API_KEY is set)
  2. nvidia (if NVIDIA_API_KEY is set)
  3. gemini (if GEMINI_API_KEY is set)
  4. ollama (local fallback)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_default_provider():
    choice = os.environ.get("LLM_PROVIDER", "")
    if not choice:
        if os.environ.get("GROQ_API_KEY"):
            choice = "groq"
        elif os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVAPI_KEY"):
            choice = "nvidia"
        elif os.environ.get("GEMINI_API_KEY"):
            choice = "gemini"
        else:
            choice = "ollama"

    if choice == "groq":
        from groq_provider import GroqProvider
        return GroqProvider()
    if choice == "nvidia":
        from nvidia_nim import NvidiaNimProvider
        return NvidiaNimProvider()
    if choice == "ollama":
        from ollama_llm import OllamaProvider
        return OllamaProvider()
    from gemini_provider import GeminiProvider
    return GeminiProvider()
