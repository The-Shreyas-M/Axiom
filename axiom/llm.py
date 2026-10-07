"""Thin Ollama client: streaming chat with optional images. Fully local."""
import base64, json, requests
from . import config as C


def available():
    try:
        r = requests.get(f"{C.OLLAMA}/api/tags", timeout=2)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def chat_stream(messages, model=None, images=None, temperature=0.2, max_tokens=700):
    """Yield text tokens. `images`: list of file paths attached to the last user message."""
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = [base64.b64encode(open(p, "rb").read()).decode() for p in images]
    body = {"model": model or C.MODEL, "messages": msgs, "stream": True, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_ctx": C.NUM_CTX, "num_predict": max_tokens}}
    try:
        with requests.post(f"{C.OLLAMA}/api/chat", json=body, stream=True, timeout=(5, 300)) as r:
            if r.status_code != 200:
                yield f"\n[LLM error {r.status_code}: {r.text[:200]}]"
                return
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if d.get("message", {}).get("content"):
                    yield d["message"]["content"]
                if d.get("done"):
                    break
    except requests.ConnectionError:
        yield "\n[Cannot reach Ollama at %s. Start it with `ollama serve`.]" % C.OLLAMA


def chat(messages, **kw):
    return "".join(chat_stream(messages, **kw))
