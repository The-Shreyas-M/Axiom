"""Provider factory. Uses Groq for text (fast) and Gemini for vision (figure captions).

TEXT_PROVIDER env overrides the text backend (default: groq).
VISION_PROVIDER env overrides the vision backend (default: gemini).
LLM_PROVIDER overrides both for backward compat.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_text_provider = None
_vision_provider = None


def _make_provider(choice: str):
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


def get_text_provider():
    global _text_provider
    if _text_provider is None:
        choice = os.environ.get("TEXT_PROVIDER") or os.environ.get("LLM_PROVIDER", "groq")
        _text_provider = _make_provider(choice)
    return _text_provider


def get_vision_provider():
    global _vision_provider
    if _vision_provider is None:
        choice = os.environ.get("VISION_PROVIDER", "gemini")
        _vision_provider = _make_provider(choice)
    return _vision_provider


def get_default_provider():
    return get_text_provider()
