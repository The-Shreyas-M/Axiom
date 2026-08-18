"""Axiom — Multimodal Research Paper Assistant.

Gradio UI with on-demand figure captioning, proper progress tracking,
session persistence, and clean layout.
"""

from __future__ import annotations

import json
import os
import time

import gradio as gr

from pipeline import SessionStore

store = SessionStore()

SESSION_DIR = os.path.join(os.path.dirname(__file__), ".axiom_session")
SESSION_META = os.path.join(SESSION_DIR, "session.json")
FIGURE_CAPTIONS: dict[str, str] = {}  # image_path -> caption


def _save_session():
    """Persist session metadata so refresh doesn't lose everything."""
    os.makedirs(SESSION_DIR, exist_ok=True)
    data = {
        "papers": [
            {
                "name": p.paper.name,
                "path": p.paper.path,
                "pages": p.paper.stats["pages"],
                "tables": p.paper.stats["tables"],
                "figures": p.paper.stats["figures"],
                "classification": p.classification,
                "ocr_pages": len(p.ocr_text_pages),
            }
            for p in store.papers
        ],
        "figures": [
            {"image": img, "caption": cap, "label": label, "paper": paper}
            for img, cap, label, paper in store.paper_figures
        ],
        "captions": FIGURE_CAPTIONS,
        "timestamp": time.time(),
    }
    with open(SESSION_META, "w") as f:
        json.dump(data, f, indent=2)


def _load_session() -> bool:
    """Try to restore session from disk."""
    if not os.path.exists(SESSION_META):
        return False
    try:
        with open(SESSION_META) as f:
            data = json.load(f)
        if time.time() - data.get("timestamp", 0) > 86400:
            return False  # too old
        global FIGURE_CAPTIONS
        FIGURE_CAPTIONS = data.get("captions", {})
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Process tab
# ---------------------------------------------------------------------------
def process_files(files, progress=gr.Progress()):
    if not files:
        return None, None, None, None
    store.reset()
    results = []
    for i, f in enumerate(files):
        pct_start = i / len(files)
        pct_end = (i + 1) / len(files)

        def file_progress(p, desc):
            progress(pct_start + p * (pct_end - pct_start), desc=desc)

        try:
            results.append(store.process_pdf(f, progress=file_progress))
        except Exception as exc:
            gallery = [(img, cap or label) for img, cap, label, _ in store.paper_figures]
            return (f"**Error:** {exc}", store.stats(), gallery, gallery)

    progress(0.85, "Generating summary...")
    try:
        summary = "\n\n".join(store.llm.summarize(p.paper.full_text) for p in results)
    except Exception as exc:
        summary = f"_Summary error: {exc}_"

    progress(1.0, "Done!")
    _save_session()
    gallery = [(img, cap or label) for img, cap, label, _ in store.paper_figures]
    return summary, store.stats(), gallery, gallery


# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------
def chat(history, message):
    if not message.strip():
        return history, []
    history = history or []
    history.append({"role": "user", "content": message})
    try:
        answer, fig_images = store.answer(message)
    except Exception as exc:
        answer = f"Error: {exc}"
        fig_images = []
    history.append({"role": "assistant", "content": answer})
    refs = []
    for img in fig_images[:6]:
        label = os.path.basename(img)
        for _, cap, lbl, _ in store.paper_figures:
            if _ == img:
                label = lbl
                break
        refs.append((img, label))
    return history, refs


# ---------------------------------------------------------------------------
# Figures tab
# ---------------------------------------------------------------------------
def on_gallery_select(evt: gr.SelectData):
    if evt.index is None or evt.index >= len(store.paper_figures):
        return None, "", "Select a figure and click **Caption** to generate a description."
    img, caption, label, paper = store.paper_figures[evt.index]
    cached = FIGURE_CAPTIONS.get(img, "")
    info = f"**{paper}** — {label}"
    if cached:
        info += f"\n\n{cached}"
    else:
        info += "\n\n_Caption not generated yet. Click Caption below._"
    return img, info, f"{paper} — {label}"


def caption_selected(image):
    """Generate caption for the selected figure on-demand."""
    if not image:
        return "Select a figure first."
    if image in FIGURE_CAPTIONS:
        return FIGURE_CAPTIONS[image]
    try:
        caption = store.caption_figure_on_demand(image)
        FIGURE_CAPTIONS[image] = caption
        _save_session()
        return caption
    except Exception as exc:
        return f"Error: {exc}"


def ask_figure(image, question):
    if not image:
        return "Select a figure from the gallery first."
    if not question.strip():
        return "Type a question about the figure."
    try:
        return store.captioner.ask_about_image(image, question.strip())
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Axiom") as demo:
    gr.Markdown(
        "# Axiom\n"
        "Upload research PDFs — extracts text, tables, figures, "
        "indexes them (SBERT + FAISS), classifies, summarizes, and answers "
        "questions with citations."
    )

    with gr.Tab("Process"):
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Upload PDF(s)", file_count="multiple",
                                file_types=[".pdf"])
                process_btn = gr.Button("Process papers", variant="primary")
            with gr.Column(scale=2):
                summary_out = gr.Markdown(label="Summary")
        with gr.Row():
            with gr.Column(scale=1):
                classify_out = gr.Markdown(label="Classification & Stats")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Extracted figures", columns=4, height=280,
                                     preview=True)

    with gr.Tab("Ask"):
        gr.Markdown("Ask anything about the processed paper(s). Answers include citations.")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(placeholder="e.g. What method do they use for attention?",
                                 show_label=False)
            with gr.Column(scale=1):
                gr.Markdown("### Referenced figures")
                ref_gallery = gr.Gallery(label="Figures", columns=2, height=480, preview=True)
        msg.submit(chat, [chatbot, msg], [chatbot, ref_gallery])
        msg.submit(lambda: "", None, msg)

    with gr.Tab("Figures"):
        gr.Markdown(
            "Select a figure, then **Caption** it or ask the vision model about it."
        )
        with gr.Row():
            with gr.Column(scale=1):
                fig_gallery = gr.Gallery(label="All figures", columns=4, height=300,
                                         preview=True)
            with gr.Column(scale=1):
                fig_image = gr.Image(label="Selected figure", type="filepath", height=300)
                fig_info = gr.Markdown()
        with gr.Row():
            caption_btn = gr.Button("Caption this figure", variant="secondary")
            fig_q = gr.Textbox(placeholder="Ask about the figure...",
                               scale=3, show_label=False)
            ask_btn = gr.Button("Ask", variant="primary", scale=1)
        fig_ans = gr.Markdown(label="Answer")

        fig_gallery.select(on_gallery_select, None, [fig_image, fig_info, fig_info])
        caption_btn.click(caption_selected, fig_image, fig_ans)
        ask_btn.click(ask_figure, [fig_image, fig_q], fig_ans)
        fig_q.submit(ask_figure, [fig_image, fig_q], fig_ans)

    process_btn.click(
        process_files,
        inputs=[files],
        outputs=[summary_out, classify_out, gallery, gallery],
    )


def main():
    _load_session()
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
