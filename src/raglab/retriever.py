"""Exact cosine search and the evidence-based hit test."""
import numpy as np

from .chunker import Chunk
from .loader import normalize


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Vectors are unit length, so the dot product equals cosine similarity."""
    return np.argsort(-(matrix @ query))[:k].tolist()


def is_hit(chunks: list[Chunk], evidence: list[list[str]]) -> bool:
    """A hit needs every evidence group found in a retrieved chunk.

    Each group lists alternative verbatim snippets, because a fact can appear in more
    than one place (for example the abstract and a results table).
    """
    texts = [normalize(c.text) for c in chunks]
    return all(any(normalize(s) in t for s in group for t in texts) for group in evidence)
