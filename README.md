# Axiom

> Multimodal research paper assistant — upload PDFs, get text, tables and figures extracted, indexed with RAG, classified, summarized, and ready for citation-backed Q&A. Even works on **scanned** papers.

Axiom turns research PDFs into a queryable multimodal corpus. It extracts text, tables and figures (including from scans, via GPU OCR + layout analysis), embeds everything into one vector space, and answers questions with page/table/figure citations — plus true vision Q&A on the raw figure images.

---

## What it does

- **Parse** PDFs with PyMuPDF: page text, `find_tables()` tables, raster figures (with captions), and vector-drawn figures (clustered drawing paths).
- **Detect scans** — a page is flagged as scanned when it has little extractable text *or* when embedded images cover ≥ 50% of the page. The second rule catches scan-strip PDFs (each page is one scan tiled into thin image strips) that still carry a baked-in OCR text layer.
- **OCR** scanned pages with **PaddleOCR** on GPU (EasyOCR fallback), producing clean reading-ordered text.
- **Layout analysis** of scans with PaddleOCR **PP-Structure**: detects `figure`/`table`/`text` regions directly on the page image, crops real figure PNGs, recovers figures the layout model misses (pure-number tables) by cropping the band above their captions, and flattens table HTML.
- **Classify** papers (NLTK stem/lemmatize → TF-IDF → logistic regression, trained on bundled arXiv samples).
- **Caption** figures with a vision model so figures live in the same retrieval space as text.
- **Embed + index** everything (text chunks, tables, figure captions) with SBERT (`bge-small-en-v1.5`, 384-dim) into a FAISS index with citation metadata.
- **Answer** questions strictly from retrieved context with page/table/figure citations, and summarize papers in a structured format.
- **Gradio UI** with three tabs: **Process** (upload → summary, classification, stats, figure gallery), **Ask** (RAG Q&A + referenced figures), **Figures** (browse + ask the vision model about any extracted figure).

## Pipeline

```
PDF
 └─ parse_pdf (pdf_parser.py)      text / tables / raster+vector figures / scanned-page flags
     └─ if scanned & OCR enabled
         ├─ OCR (ocr.py)           PaddleOCR GPU → reading-ordered page text
         └─ layout (layout.py)     PP-Structure → figure crops + table HTML
 └─ classifier.py                  category prediction
 └─ vision.py                      figure captions (Gemini / Ollama)
 └─ embeddings.py + retriever.py   SBERT chunks → FAISS index (citations)
 └─ llm.py                         RAG answers + structured summaries
```

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Gradio UI (Process / Ask / Figures tabs) |
| `pdf_parser.py` | PDF parsing: text, tables, raster & vector figures, scanned-page detection |
| `ocr.py` | Multi-engine OCR facade (PaddleOCR default, EasyOCR fallback), NVIDIA DLL path handling |
| `layout.py` | PP-Structure layout analysis for scanned pages (figure/table extraction) |
| `pipeline.py` | `SessionStore`: parse → OCR/layout → classify → caption → embed → index → answer |
| `classifier.py` | TF-IDF + logistic-regression paper classifier |
| `embeddings.py` | SBERT embedder + `chunk_text` |
| `retriever.py` | FAISS index with citation metadata |
| `llm.py` | Citation-aware summaries and RAG answers |
| `vision.py` | Figure captioning + figure Q&A via vision model |
| `providers.py` | LLM provider factory (Gemini / Ollama) |
| `gemini_provider.py` | Google Gemini text + vision client (Interactions API) |
| `ollama_llm.py` | Local Ollama text + vision client |
| `data/arxiv_samples.json` | Training data for the classifier |
| `requirements.txt` | Base dependencies |
| `requirements-ocr.txt` | Locked OCR stack (paddleocr + paddlepaddle-gpu + NVIDIA DLLs) |
| `colab_setup.ipynb` | Run-on-Colab launcher |
| `Dockerfile` | Container image |

## Getting started

Requires Python 3.12.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\pip install -r requirements.txt
# macOS/Linux:
source .venv/bin/activate && pip install -r requirements.txt
```

### Optional: OCR + layout analysis for scanned PDFs

```bash
.venv\Scripts\pip install -r requirements-ocr.txt
```

The locked stack (paddleocr 2.9.1 + paddlepaddle-gpu 2.6.2 + `nvidia-cudnn-cu12` / `nvidia-cublas-cu11` / `nvidia-cuda-runtime-*`) is pinned because newer combinations crash on Windows (see [Engineering notes](#engineering-notes)). Verified on an NVIDIA RTX 3050 6 GB.

### Configure the model backend

Create a `.env` in the project root:

```
GEMINI_API_KEY=your_key_here
```

A free key: <https://aistudio.google.com/apikey>

- No key set → falls back to local **Ollama** (`ollama serve`, e.g. `ollama pull gemma3:4b`).
- `LLM_PROVIDER=gemini` or `LLM_PROVIDER=ollama` overrides auto-detection.
- `LLM_MODEL` and `OLLAMA_BASE_URL` override model / server.

### Run

```bash
.venv\Scripts\python app.py
```

Open <http://127.0.0.1:7860>. Upload PDFs on the **Process** tab, tick **OCR scanned pages (paddleocr)** for scans, then ask questions on the **Ask** tab.

Environment variables:

| Var | Default | Purpose |
| --- | --- | --- |
| `GRADIO_SHARE` | `false` | `true` exposes a public Gradio share link |
| `LLM_PROVIDER` | auto | `gemini` or `ollama` |
| `GEMINI_API_KEY` | — | Gemini key (`.env` or env) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model name |
| `LLM_MODEL` | `gemma3:4b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |

## Docker

```bash
docker build -t axiom .
docker run -p 7860:7860 --env-file .env axiom
```

## Colab

Open `colab_setup.ipynb`, set your `GEMINI_API_KEY` as a Colab secret, and run the cells — the last one prints a public Gradio link.

## Engineering notes

The messy corners of making OCR actually work on this machine, in case anyone repeats it:

- **Windows `paddlepaddle-gpu` doesn't bundle CUDA/cuDNN runtimes.** The 2.6.2 wheel is a CUDA 11.8 build that needs `cudnn64_8.dll` / `cublas64_11.dll` / CUDA runtime. Axiom pulls those from the pip packages `nvidia-cudnn-cu12`, `nvidia-cublas-cu11`, `nvidia-cuda-runtime-cu11/cu12` and injects their `bin`/`lib` dirs into `PATH` at runtime (`ocr._add_nvidia_paths()`). This is also required by PP-Structure.
- **PaddleOCR version lock.** `paddleocr 3.x + paddlepaddle 3.x` segfaults / throws `NotImplementedError` (oneDNN) on this setup. The working combo is **`paddleocr==2.9.1` + `paddlepaddle-gpu==2.6.2`**, with `numpy<2.0` and `protobuf==3.20.2` pinned.
- **Scanned-page detection.** Text-length alone misses scans that already have an OCR text layer baked in (common on archive scans). Axiom also flags pages whose placed images cover ≥ 50% of the page — this catches scan-strip PDFs where each page is a full-page scan tiled into thin strips.
- **Figures from scans.** PyMuPDF can't find structure inside a scan, so `layout.py` runs PP-Structure on the page image. The layout detector skips pure-number tables (relational data grids), so Axiom recovers them by cropping the band between a figure's caption and the layout region above it, bounding the column width with neighboring regions. Caption matching tolerates OCR noise (`FIG` → `Fia`/`FIg`/`F1G`).
- **BGE retrieval instruction.** bge models recommend a query-side prefix (`Represent this sentence for searching relevant passages: `), applied to queries only.

## Status / roadmap

See [TODO.md](TODO.md).
