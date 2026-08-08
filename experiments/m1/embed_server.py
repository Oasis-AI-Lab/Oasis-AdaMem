# Local CPU OpenAI-compatible embedding service (Phase 2)
# Usage: .venv/bin/python experiments/m1/embed_server.py [--port 8002] [--model Qwen/Qwen3-Embedding-0.6B]
# Compatible with utils.py: POST /v1/embeddings {"model": ..., "input": [...]} -> {"data": [{"embedding": [...], "index": i}]}
import argparse
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # China mirror
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class EmbedRequest(BaseModel):
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    input: list | str

class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str = "list"

_model = None
_model_name = ""

# Model alias mapping: official code hardcodes requests for Qwen3-Embedding-4B
# (8GB, not feasible on CPU); this service maps it to the same-family 0.6B (dim 1024).
# Official code stays untouched.
MODEL_ALIASES = {
    "Qwen/Qwen3-Embedding-4B": "Qwen/Qwen3-Embedding-0.6B",
}


def _resolve_model(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def _load_model(model_name: str):
    global _model, _model_name
    model_name = _resolve_model(model_name)
    if _model is not None and _model_name == model_name:
        return
    from sentence_transformers import SentenceTransformer
    print(f"[embed_server] loading {model_name} (CPU) ...", flush=True)
    t0 = time.time()
    _model = SentenceTransformer(model_name, device="cpu")
    _model_name = model_name
    print(f"[embed_server] loaded in {time.time()-t0:.1f}s", flush=True)


@app.post("/v1/embeddings")
def embed(req: EmbedRequest):
    _load_model(req.model)
    texts = req.input if isinstance(req.input, list) else [req.input]
    t0 = time.time()
    # prompt=0: no instruction prefix (the caller owns the get_detailed_instruct prefix, matching the paper)
    embs = _model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    data = [{"embedding": embs[i].tolist(), "index": i} for i in range(len(texts))]
    print(f"[embed_server] embedded {len(texts)} texts in {time.time()-t0:.2f}s", flush=True)
    return EmbeddingResponse(data=data, model=req.model)


@app.get("/v1/models")
def models():
    return {"data": [{"id": _model_name or "Qwen/Qwen3-Embedding-0.6B"}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    args = ap.parse_args()
    _load_model(args.model)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
