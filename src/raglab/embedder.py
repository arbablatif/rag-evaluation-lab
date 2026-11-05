"""Local sentence-transformers embeddings, cached on disk per (model, chunk size)."""
import json
import time
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .chunker import Chunk
from .config import CACHE_DIR, EMBEDDERS


@lru_cache(maxsize=2)
def load_model(key: str) -> SentenceTransformer:
    # Force float32: some checkpoints default to bfloat16, which is far slower on CPU and
    # would distort the latency comparison between models.
    return SentenceTransformer(EMBEDDERS[key], device="cpu",
                               model_kwargs={"dtype": torch.float32})


def unread_share(key: str, chunks: list[Chunk]) -> float:
    """Average share of each chunk's tokens past the model's limit, which it never reads."""
    model = load_model(key)
    limit = model.max_seq_length
    lengths = [len(model.tokenizer(c.text, add_special_tokens=True, truncation=False)["input_ids"])
               for c in chunks]
    return float(np.mean([max(0, n - limit) / n for n in lengths]))


def embed_chunks(key: str, size: int, chunks: list[Chunk]) -> tuple[np.ndarray, dict]:
    """Return unit-length chunk vectors plus index metadata (build time, truncation)."""
    folder = CACHE_DIR / "embeddings-fp32"  # precision is part of the key: vectors differ by it
    folder.mkdir(parents=True, exist_ok=True)
    vec_path, meta_path = folder / f"{key}-{size}.npy", folder / f"{key}-{size}.json"
    if vec_path.exists() and meta_path.exists():
        return np.load(vec_path), json.loads(meta_path.read_text())

    model = load_model(key)
    start = time.perf_counter()
    vectors = model.encode([c.text for c in chunks], batch_size=8, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    meta = {"index_seconds": round(time.perf_counter() - start, 2),
            "n_chunks": len(chunks),
            "max_seq_length": model.max_seq_length}
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps(meta))
    return vectors, meta


def embed_query(key: str, question: str) -> np.ndarray:
    return load_model(key).encode(question, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
