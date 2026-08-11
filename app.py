"""Gradio UI: Process / Ask / Figures tabs for the Multimodal Paper Assistant."""

from __future__ import annotations

import os
import time

import gradio as gr

from pipeline import SessionStore

store = SessionStore()

PLACEHOLDER_ANSWER = (
    "I couldn't retrieve relevant content. Have you processed a paper first? "
    "(If the model backend isn't configured, set GEMINI_API_KEY in a .env file "
    "or start Ollama with `ollama serve`.)"
)


# ---------------------------------------------------------------------------
# Process tab
# ---------------------------------------------------------------------------
def process_files(files, use_ocr: bool, progress=gr.Progress()):
    if not files:
        return None, None, None
    results = []
    for f in files:
        try:
            results.append(store.process_pdf(f, use_ocr=use_ocr, progress=progress))
        except Exception as exc:
            return (f"**Error processing {os.path.basename(f)}:** {exc}",
                    None, None)

    try:
        summary = "\n\n".join(
            store.llm.summarize(p.paper.full_text) for p in results
        )
    except Exception as exc:
        summary = f"_Summary skipped: {exc}_"
    gallery = [(img, cap or label) for img, cap, label, _ in store.paper_figures]
    return summary, store.stats(), gallery


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
    return history, [(img, os.path.basename(os.path.dirname(img)) + " / " + os.path.basename(img))
                     for img in fig_images]


# ---------------------------------------------------------------------------
# Figures tab
# ---------------------------------------------------------------------------
def on_gallery_select(evt: gr.SelectData):
    if evt.index is None or evt.index >= len(store.paper_figures):
        return None, "", ""
    img, caption, label, paper = store.paper_figures[evt.index]
    return img, caption, f"{paper} — {label}"


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
# Build UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Multimodal Research Paper Assistant") as demo:
    gr.Markdown(
        "# Multimodal Research Paper Assistant\n"
        "Upload research PDFs — the assistant extracts text, tables and figures, "
        "indexes them (BERT + RAG), classifies the paper, summarizes it, and answers "
        "questions with citations, including questions about figures themselves."
    )

    with gr.Tab("Process"):
        with gr.Row():
            with gr.Column():
                files = gr.File(label="Upload research PDF(s)", file_count="multiple",
                                file_types=[".pdf"])
                use_ocr = gr.Checkbox(label="OCR scanned pages (paddleocr)", value=False)
                process_btn = gr.Button("Process papers", variant="primary")
            with gr.Column():
                summary_out = gr.Markdown(label="Summary")
        with gr.Row():
            with gr.Column(scale=1):
                classify_out = gr.Markdown(label="Classification & Stats")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Extracted figures", columns=4, height=280)

        process_btn.click(
            process_files,
            inputs=[files, use_ocr],
            outputs=[summary_out, classify_out, gallery],
        )

    with gr.Tab("Ask"):
        gr.Markdown("Ask anything about the processed paper(s) — text, tables or figures.")
        chatbot = gr.Chatbot(height=420)
        msg = gr.Textbox(placeholder="e.g. What method do they use for attention?")
        with gr.Row():
            gr.Markdown("### Referenced figures (from retrieved chunks)")
        ref_gallery = gr.Gallery(label="Referenced figures", columns=4, height=240)
        msg.submit(chat, [chatbot, msg], [chatbot, ref_gallery])
        msg.submit(lambda: "", None, msg)

    with gr.Tab("Figures"):
        gr.Markdown(
            "Select a figure from the gallery, then ask the vision model directly "
            "about it — this is true multimodal Q&A on the raw image."
        )
        with gr.Row():
            fig_gallery = gr.Gallery(label="All extracted figures", columns=4, height=280)
            with gr.Column():
                fig_image = gr.Image(label="Selected figure", type="filepath")
                fig_info = gr.Markdown()
        fig_q = gr.Textbox(placeholder="Ask about the selected figure, e.g. What trend does this plot show?")
        fig_ans = gr.Markdown(label="Vision answer")
        fig_gallery.select(on_gallery_select, None, [fig_image, fig_info, fig_info])
        fig_q.submit(ask_figure, [fig_image, fig_q], fig_ans)


def main():
    # Local by default; set GRADIO_SHARE=true to expose via a public link.
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
