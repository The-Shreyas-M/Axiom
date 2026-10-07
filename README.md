# Axiom — offline multimodal research-paper assistant

Drop in a research-paper PDF (digital **or scanned**) and Axiom pulls it apart into **text, figures, tables and captions**, indexes every piece separately, writes a summary, and lets you **chat with the paper — or with one specific figure, table or paragraph**.

Everything runs **locally**: no accounts, no API keys, no cloud, no GPU required.

![stack](https://img.shields.io/badge/runs-100%25%20offline-4f46e5) ![py](https://img.shields.io/badge/python-3.10%2B-blue) ![gpu](https://img.shields.io/badge/GPU-optional-green)

---

## Quick start (one click)

**Windows**

1. Install [Python 3.10+](https://www.python.org/downloads/) (tick *Add Python to PATH*).
2. Double-click **`run.bat`**.

That's it. On the first run it will, automatically:

| Step | What happens | Needs internet? |
|---|---|---|
| 1 | Installs [Ollama](https://ollama.com) via `winget` if missing | yes (once) |
| 2 | Creates a virtual env and installs the Python packages | yes (once) |
| 3 | Downloads the embedding model (~130 MB) and the `gemma3:4b` model (~3.3 GB) | yes (once) |
| 4 | Starts the app and opens **http://127.0.0.1:8000** | no |

Every run after that is fully offline and starts in seconds. **Use the red ⏻ Quit Axiom button in the sidebar to stop it** — no Ctrl+C needed.

**macOS / Linux:** install Python and [Ollama](https://ollama.com/download), then `./run.sh`.

---

## Using it

1. **Upload** a PDF (drag onto the left panel). A progress bar shows parsing → embedding. A 55-page paper takes roughly 20 s on a laptop.
2. **Summary tab** – a structured summary (TL;DR, problem, approach, key results, limitations) starts streaming as soon as the paper is parsed.
3. **Elements tab** – every extracted element as a card: filter by *text / figure / table / caption*, search, view the cropped image, caption, table contents and an AI "visual summary".
   - **Ask about this** → focuses the chat on that exact element (the image is sent to the vision model).
   - **Analyse image** → generates a description of a figure on demand.
   - **View page** → jumps to the page in the original PDF.
4. **Ask the paper** (right panel) – ask anything. Answers stream live and cite sources like `[p3]`, `[Figure 2]`, `[Table 1]`. Click a source chip to jump to that element.
   - Mention things naturally: *"What does Figure 3 show?"*, *"Compare Table 2 and Table 3"*, *"What's on page 7?"* — those references are resolved directly, not guessed.

---

## How it works

```
 PDF ──► Parse ──────────────► Elements ───► Embed + index ───► Hybrid retrieval ───► Local LLM / VLM
         │                      text chunks    bge-small (ONNX)   dense (Chroma)        Ollama
         │ digital: PyMuPDF     figures        Chroma on disk     + BM25 keywords       gemma3:4b
         │ scanned: RapidOCR    tables         one collection     + explicit            (text + vision)
         │          + OpenCV    captions       per paper            "Figure N" lookup
         ▼
   section tracking, 2-column reading order, caption↔figure linking
```

**Parsing** (`axiom/parser.py`)
- *Digital pages:* PyMuPDF text blocks with font stats → heading/section detection, two-column reading order, de-hyphenation. Figures = embedded images + clustered vector drawings; tables = PyMuPDF table finder (only run on pages with a "Table N" caption — it is the slowest step) with a fallback that rebuilds tables PyMuPDF misses from the text/rules under the caption.
- *Scanned pages* (almost no text layer): page is rendered, OCR'd with RapidOCR, OCR lines are grouped into paragraphs, and OpenCV finds figure regions (non-text ink) and tables (ruling lines).
- *Captions* (`Figure 3:`, `Fig. 3—`, `Table 2.`) are stored as their own elements **and** attached to the figure/table they describe.

**Keeping context** — a figure's index entry is its label + caption + the sentences elsewhere in the paper that mention it ("As Figure 3 shows…") + text inside it + a VLM description. Text chunks carry their section name. So a question about a figure finds the right figure even if the caption is terse.

**Retrieval** (`axiom/store.py`) — dense search (Chroma, cosine) fused with BM25 via reciprocal-rank fusion; explicit `Figure N` / `Table N` / `page N` in the question are resolved exactly; the element you clicked is always included first.

**Speed design** — the summary is generated once in the background and streamed; figure descriptions are generated in the background *at lowest priority and are interrupted the instant you ask a question*; the LLM and embedder are pre-warmed at startup; embeddings use length-sorted micro-batches (≈3× faster on CPU); prompts are kept small.

---

## Project layout

```
axiom/
  server.py     FastAPI app + endpoints (upload, ask [SSE], summary [SSE], shutdown)
  parser.py     PDF → elements (digital + scanned)
  ingest.py     pipeline orchestration, background figure descriptions, title detection
  store.py      embeddings, Chroma index, BM25, hybrid retriever
  qa.py         grounded prompts, streaming answers, summary generation
  llm.py        tiny streaming Ollama client
  config.py     all settings (env-overridable)
  setup.py      one-time model downloader (called by run.bat / run.sh)
  static/index.html   the whole UI (no build step)
run.bat / run.sh      one-click launchers
legacy/               the earlier Gradio/FAISS prototype (kept for reference)
data/                 created at runtime: papers, images, vector DB, models (git-ignored)
```

---

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `AXIOM_MODEL` | `gemma3:4b` | Ollama model for text answers & summary |
| `AXIOM_VLM` | same as above | Ollama model for image questions / figure descriptions (must be multimodal) |
| `AXIOM_NUM_CTX` | `4096` | LLM context window (lower = faster, less VRAM) |
| `AXIOM_AUTODESCRIBE` | `1` | `0` disables background figure descriptions |
| `AXIOM_DATA` | `./data` | where papers/index/models are stored |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `AXIOM_ONLINE` | unset | `1` re-enables network (offline mode is forced once models are cached) |

**Choosing a model for your hardware**

| Machine | Suggestion |
|---|---|
| ≥6 GB GPU | default `gemma3:4b` — ~4 s answers |
| CPU only / low RAM | `set AXIOM_MODEL=llama3.2:3b` and `set AXIOM_AUTODESCRIBE=0` (text answers stay fast; use *Analyse image* on demand) |
| Strong GPU | `gemma3:12b` for better figure understanding |

Example: `set AXIOM_MODEL=gemma3:12b && run.bat`

---

## Troubleshooting

- **"Ollama not running" in the sidebar** – start it (`ollama serve`) or reopen the Ollama app.
- **"Model missing"** – `ollama pull gemma3:4b`.
- **First question is slow** – the model is loading into memory once; later answers are fast.
- **Re-run first-time setup** – delete `data/.setup_done` and run `run.bat` again.
- **Move to another machine offline** – copy the whole folder (including `data/models`) and run `ollama pull gemma3:4b` once on the target.

## Known limitations

- Math/equations are extracted as plain text, not LaTeX.
- Complex multi-page tables and figures that are pure vector text may be missed or split.
- Scanned pages cost ~9 s/page on CPU (OCR), much less on fast CPUs.
- Answer quality is bounded by a 4B local model — it cites sources so you can verify.

## Tech

PyMuPDF · RapidOCR (ONNX) · OpenCV · fastembed (bge-small, ONNX) · ChromaDB · rank-bm25 · FastAPI · Ollama (gemma3) · vanilla HTML/JS.
