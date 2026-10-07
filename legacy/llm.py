"""LLM layer: citation-aware summaries and RAG answers. Uses Groq (fast)."""

from __future__ import annotations

from providers import get_text_provider

SUMMARIZE_SYSTEM = (
    "Summarize this research paper in markdown with sections: "
    "**TL;DR**, **Problem**, **Method**, **Key Results**, "
    "**Limitations**, **Main Contributions**. Be concise."
)

ANSWER_SYSTEM = (
    "Answer from the context below. Cite sources like (p.4) or (Fig. 3). "
    "If the context lacks the answer, say so. Be concise."
)

MAX_SUMMARY_CHARS = 28_000


class ResearchLLM:
    def __init__(self, provider=None):
        self.provider = provider or get_text_provider()

    def summarize(self, full_text: str) -> str:
        text = (full_text or "").strip()
        if not text:
            return "_No extractable text._"
        return self.provider.generate_text(text[:MAX_SUMMARY_CHARS], system=SUMMARIZE_SYSTEM)

    def answer(self, question: str, hits: list[dict]) -> str:
        if not hits:
            return "No relevant content found."
        blocks = []
        for i, hit in enumerate(hits, 1):
            blocks.append(f"[{i}] {hit['label']} (p.{hit['page']}, {hit['type']})\n{hit['text'][:1000]}")
        prompt = "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
        return self.provider.generate_text(prompt, system=ANSWER_SYSTEM)
