"""Axiom — Multimodal Research Paper Assistant."""

from __future__ import annotations

import os

import gradio as gr

from pipeline import SessionStore

store = SessionStore()
store.load()


def process_files(files, progress=gr.Progress()):
    if not files:
        return None, None, None, None
    store.reset()
    results = []
    for i, f in enumerate(files):
        pct0 = i / len(files)
        pct1 = (i + 1) / len(files)

        def file_progress(p, desc):
            progress(pct0 + p * (pct1 - pct0), desc=desc)

        try:
            results.append(store.process_pdf(f, progress=file_progress))
        except Exception as exc:
            gallery = [(img, cap or label) for img, cap, label, _ in store.paper_figures]
            return (f"**Error:** {exc}", store.stats(), gallery, gallery)

    progress(0.88, "Summarizing...")
    try:
        summary = "\n\n".join(store.llm.summarize(p.paper.full_text) for p in results)
    except Exception as exc:
        summary = f"_Summary error: {exc}_"

    progress(1.0, "Done!")
    gallery = [(img, cap or label) for img, cap, label, _ in store.paper_figures]
    return summary, store.stats(), gallery, gallery


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
        for tup in store.paper_figures:
            if tup[0] == img:
                label = tup[2]
                break
        refs.append((img, label))
    return history, refs


def on_gallery_select(evt: gr.SelectData):
    if evt.index is None or evt.index >= len(store.paper_figures):
        return None, ""
    img, caption, label, paper = store.paper_figures[evt.index]
    return img, f"**{paper}** — {label}\n\n{caption or '_No caption yet. Click Caption._'}"


def caption_selected(image):
    if not image:
        return "Select a figure first."
    try:
        caption = store.caption_figure_on_demand(image)
        for i, (img, cap, label, paper) in enumerate(store.paper_figures):
            if img == image:
                store.paper_figures[i] = (img, caption, label, paper)
                break
        store.save()
        return caption
    except Exception as exc:
        return f"Error: {exc}"


def ask_figure(image, question):
    if not image:
        return "Select a figure first."
    if not question.strip():
        return "Type a question."
    try:
        return store.captioner.ask_about_image(image, question.strip())
    except Exception as exc:
        return f"Error: {exc}"


with gr.Blocks(title="Axiom") as demo:
    gr.Markdown("# Axiom\nUpload research PDFs — extracts text, tables, figures, indexes them, classifies, summarizes, and answers with citations.")

    with gr.Tab("Process"):
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Upload PDF(s)", file_count="multiple", file_types=[".pdf"])
                process_btn = gr.Button("Process papers", variant="primary")
            with gr.Column(scale=2):
                summary_out = gr.Markdown(label="Summary")
        with gr.Row():
            with gr.Column(scale=1):
                classify_out = gr.Markdown(label="Classification & Stats")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Figures", columns=4, height=280, preview=True)

    with gr.Tab("Ask"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(placeholder="e.g. What method do they use?", show_label=False)
            with gr.Column(scale=1):
                gr.Markdown("### Referenced figures")
                ref_gallery = gr.Gallery(columns=2, height=480, preview=True)
        msg.submit(chat, [chatbot, msg], [chatbot, ref_gallery])
        msg.submit(lambda: "", None, msg)

    with gr.Tab("Figures"):
        with gr.Row():
            with gr.Column(scale=1):
                fig_gallery = gr.Gallery(label="Figures", columns=4, height=300, preview=True)
            with gr.Column(scale=1):
                fig_image = gr.Image(label="Selected", type="filepath", height=300)
                fig_info = gr.Markdown()
        with gr.Row():
            caption_btn = gr.Button("Caption this figure", variant="secondary")
            fig_q = gr.Textbox(placeholder="Ask about the figure...", show_label=False, scale=3)
            ask_btn = gr.Button("Ask", variant="primary", scale=1)
        fig_ans = gr.Markdown()

        fig_gallery.select(on_gallery_select, None, [fig_image, fig_info])
        caption_btn.click(caption_selected, fig_image, fig_ans)
        ask_btn.click(ask_figure, [fig_image, fig_q], fig_ans)
        fig_q.submit(ask_figure, [fig_image, fig_q], fig_ans)

    process_btn.click(process_files, inputs=[files], outputs=[summary_out, classify_out, gallery, gallery])


def main():
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
