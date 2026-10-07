import json, threading, uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as C, ingest, llm, qa, store

app = FastAPI(title="Axiom")
jobs = {}
STATIC = Path(__file__).parent / "static"


@app.on_event("startup")
def prewarm():
    """Load the embedder + LLM into memory now so the first question is not a cold start."""
    def go():
        try:
            store.embed(["warm"])
            import requests
            requests.post(f"{C.OLLAMA}/api/generate", json={"model": C.MODEL, "keep_alive": "30m",
                          "options": {"num_ctx": C.NUM_CTX}}, timeout=120)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


@app.get("/api/health")
def health():
    models = llm.available()
    return {"ollama": models is not None, "models": models or [], "model": C.MODEL,
            "model_ready": bool(models) and any(m.split(":")[0] == C.MODEL.split(":")[0] for m in models)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data[:5] == b"%PDF-":
        raise HTTPException(400, "Not a PDF")
    jid = uuid.uuid4().hex[:8]
    jobs[jid] = {"progress": 0.0, "message": "Queued", "done": False, "paper": None, "error": None}

    def run():
        try:
            def prog(p, m):
                jobs[jid].update(progress=round(p, 3), message=m)
            jobs[jid]["paper"] = ingest.ingest(data, file.filename, prog)
        except Exception as ex:  # surface to UI
            jobs[jid]["error"] = f"{type(ex).__name__}: {ex}"
        jobs[jid]["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    return jobs[jid]


@app.get("/api/papers")
def papers():
    return ingest.list_papers()


@app.get("/api/papers/{pid}")
def paper(pid: str):
    try:
        meta, els = ingest.load(pid)
    except FileNotFoundError:
        raise HTTPException(404)
    return {"meta": meta, "elements": els}


@app.delete("/api/papers/{pid}")
def delete(pid: str):
    ingest.delete(pid)
    qa.invalidate(pid)
    return {"ok": True}


@app.get("/api/papers/{pid}/file/{name:path}")
def pfile(pid: str, name: str):
    p = (ingest.paper_dir(pid) / name).resolve()
    if ingest.paper_dir(pid).resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


class Ask(BaseModel):
    question: str
    focus: str | None = None
    history: list[dict] = []


def sse(gen):
    def it():
        for ev in gen:
            yield f"data: {json.dumps(ev)}\n\n"
        yield "data: {\"done\": true}\n\n"
    return StreamingResponse(it(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/papers/{pid}/ask")
def ask(pid: str, body: Ask):
    return sse(qa.answer_stream(pid, body.question, body.focus, body.history))


@app.get("/api/papers/{pid}/summary")
def summary(pid: str):
    return sse(qa.summary_stream(pid))


@app.post("/api/papers/{pid}/describe/{eid}")
def describe(pid: str, eid: str):
    meta, els = ingest.load(pid)
    e = next((x for x in els if x["id"] == eid), None)
    if not e or not e.get("image"):
        raise HTTPException(404)
    ingest.interactive_start()
    try:
        with ingest._desc_lock:
            meta, els = ingest.load(pid)
            e = next(x for x in els if x["id"] == eid)
            if not ingest.describe_element(pid, e):
                raise HTTPException(503, "Ollama unavailable")
            ingest.save_elements(pid, els)
            store.reindex_one(pid, e)
    finally:
        ingest.interactive_end()
    qa.invalidate(pid)
    return e


@app.post("/api/shutdown")
def shutdown():
    import os
    threading.Timer(0.4, lambda: os._exit(0)).start()
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn, webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
