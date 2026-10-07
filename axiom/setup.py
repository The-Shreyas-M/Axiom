"""One-time setup: fetch the embedding model + pull the Ollama model. Safe to re-run (idempotent)."""
import json
import os
import sys

os.environ["AXIOM_ONLINE"] = "1"  # allow network for this step only
import requests

from . import config as C


def ensure_embedder():
    from fastembed import TextEmbedding
    print(f"[1/2] Embedding model {C.EMBED_MODEL} ...", flush=True)
    list(TextEmbedding(C.EMBED_MODEL, cache_dir=str(C.DATA / "models")).embed(["warm"]))
    print("      ready.")


def ensure_llm():
    print(f"[2/2] Language/vision model {C.MODEL} via Ollama ...", flush=True)
    try:
        have = [m["name"] for m in requests.get(f"{C.OLLAMA}/api/tags", timeout=3).json()["models"]]
    except Exception:
        print("      Ollama is not running. Install it from https://ollama.com/download and start it, then re-run.")
        return False
    if any(n == C.MODEL or n.split(":")[0] == C.MODEL for n in have):
        print("      already installed.")
        return True
    last = ""
    with requests.post(f"{C.OLLAMA}/api/pull", json={"model": C.MODEL, "stream": True}, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            msg = d.get("status", "")
            if d.get("total"):
                msg += f" {d.get('completed', 0) * 100 // d['total']}%"
            if msg != last:
                print("      " + msg, end="\r", flush=True)
                last = msg
    print("\n      done.")
    return True


if __name__ == "__main__":
    ensure_embedder()
    sys.exit(0 if ensure_llm() else 1)
