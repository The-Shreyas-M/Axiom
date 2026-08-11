"""LLM layer: citation-aware summaries and RAG answers. Provider-agnostic."""

from __future__ import annotations

from providers import get_default_provider

SUMMARIZE_SYSTEM = (
    "You are an expert research assistant. Write a structured summary of the "
    "following research paper. Use markdown with sections: **TL;DR**, "
    "**Problem**, **Method**, **Key Results**, **Limitations**, "
    "**Main Contributions**. Be accurate and concise."
)

SECTION_SUMMARIZE_PROMPT = (
    "This is a section of a research paper. Summarize the key technical "
    "content (methods, results, claims) in under 120 words, keeping all "
    "specific numbers and names."
)

MERGE_SUMMARY_PROMPT = (
    "Combine these section summaries into one coherent structured markdown "
    "summary with sections: **TL;DR**, **Problem**, **Method**, "
    "**Key Results**, **Limitations**, **Main Contributions**."
)

ANSWER_SYSTEM = (
    "You are a research assistant that answers questions strictly from the "
    "provided context chunks retrieved from a research paper. "
    "Rules:\n"
    "1. Answer using ONLY the context below.\n"
    "2. Cite sources inline like (p.4) for paragraphs and (Fig. 3) or "
    "(Table 2) for figures and tables, using the citation label given for "
    "each chunk.\n"
    "3. If the context does not contain the answer, say 'The paper does not "
    "appear to cover this' and do not guess.\n"
    "4. Keep the answer focused; use bullets when helpful."
)

MAX_SINGLE_SHOT_CHARS = 500_000
SECTION_CHARS = 8_000
MAX_SECTIONS = 5


def _split_for_map_reduce(text: str) -> list[str]:
    parts, rest = [], text
    while rest and len(parts) < MAX_SECTIONS:
        parts.append(rest[:SECTION_CHARS])
        rest = rest[SECTION_CHARS:]
    if rest:
        parts.append(rest)
    return parts


class ResearchLLM:
    """Local-model-powered summary + citation-aware RAG answering."""

    def __init__(self, provider=None):
        self.provider = provider or get_default_provider()

    def summarize(self, full_text: str) -> str:
        text = full_text.strip()
        if not text:
            return "_No extractable text in this paper._"
        limit = getattr(self.provider, "max_context_chars", MAX_SINGLE_SHOT_CHARS)
        if len(text) <= limit:
            return self.provider.generate_text(text, system=SUMMARIZE_SYSTEM)

        # map-reduce for long papers (local context windows are smaller)
        sections = _split_for_map_reduce(text)
        partials = [
            self.provider.generate_text(s, system=SECTION_SUMMARIZE_PROMPT)
            for s in sections
        ]
        joined = "\n\n".join(f"### Section {i+1}\n{p}" for i, p in enumerate(partials))
        return self.provider.generate_text(joined, system=MERGE_SUMMARY_PROMPT)

    def answer(self, question: str, hits: list[dict]) -> str:
        if not hits:
            return "No relevant content found in the indexed paper(s)."
        blocks = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{i}] {hit['label']} (page {hit['page']}, {hit['type']})\n"
                f"{hit['text'][:1800]}"
            )
        context = "\n\n".join(blocks)
        prompt = (
            f"Context chunks:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with inline citations matching the [i] labels above, "
            "e.g. (p.4) or (Fig. 3)."
        )
        return self.provider.generate_text(prompt, system=ANSWER_SYSTEM)
