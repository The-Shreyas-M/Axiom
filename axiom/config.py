import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AXIOM_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# One multimodal model for text + vision keeps a single model resident (fast, fits 6GB VRAM).
MODEL = os.environ.get("AXIOM_MODEL", "gemma3:4b")
VLM = os.environ.get("AXIOM_VLM", MODEL)
NUM_CTX = int(os.environ.get("AXIOM_NUM_CTX", "4096"))
EMBED_MODEL = os.environ.get("AXIOM_EMBED", "BAAI/bge-small-en-v1.5")
DESCRIBE_IN_BACKGROUND = os.environ.get("AXIOM_AUTODESCRIBE", "1") == "1"

# Once models are cached in data/models, never touch the network (set AXIOM_ONLINE=1 for first-time download).
if os.environ.get("AXIOM_ONLINE") != "1" and (DATA / "models").exists() and any((DATA / "models").iterdir()):
    os.environ["HF_HUB_OFFLINE"] = "1"
