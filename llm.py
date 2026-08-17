"""LLM layer: citation-aware summaries and RAG answers. Provider-agnostic."""

from __future__ import annotations

from providers import get_default_provider

SUMMARIZE_SYSTEM = (
    "Summarize this research paper in markdown with sections: "
    "**TL;DR**, **Problem**, **Method**, **Key Results**, "
    "**Limitations**, **Main Contributions**. Be concise and accurate."
)

ANSWER_SYSTEM = (
    "Answer from the context below. Cite sources inline like (p.4) or "
    "(Fig. 3). If the context lacks the answer, say so. Be concise."
)

MAX_SUMMARY_CHARS = 80_000


class ResearchLLM:
    """Provider-powered summary + citation-aware RAG answering."""

    def __init__(self, provider=None):
        self.provider = provider or get_default_provider()

    def summarize(self, full_text: str) -> str:
        text = full_text.strip()
        if not text:
            return "_No extractable text._"
        text = text[:MAX_SUMMARY_CHARS]
        return self.provider.generate_text(text, system=SUMMARIZE_SYSTEM)

    def answer(self, question: str, hits: list[dict]) -> str:
        if not hits:
            return "No relevant content found."
        blocks = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{i}] {hit['label']} (page {hit['page']}, {hit['type']})\n"
                f"{hit['text'][:1200]}"
            )
        context = "\n\n".join(blocks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self.provider.generate_text(prompt, system=ANSWER_SYSTEM)
