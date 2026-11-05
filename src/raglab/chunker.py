"""Token-based sliding-window chunker.

Chunks may cross page boundaries on purpose. A page holds roughly 700-1000 tokens, so
stopping at page ends would stop 1024-token chunks from ever reaching their target size.
"""
from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer

from .config import CHUNK_TOKENIZER, OVERLAP_RATIO
from .loader import Document


@dataclass(frozen=True)
class Chunk:
    doc: str
    index: int
    text: str
    pages: tuple[int, ...]
    n_tokens: int


@lru_cache(maxsize=1)
def tokenizer():
    tok = AutoTokenizer.from_pretrained(CHUNK_TOKENIZER)
    tok.model_max_length = 10**9  # we count tokens ourselves; silences the length warning
    return tok


def chunk_document(doc: Document, size: int, overlap_ratio: float = OVERLAP_RATIO) -> list[Chunk]:
    """Slide a window of `size` tokens over the document and slice the original text."""
    offsets = tokenizer()(doc.text, add_special_tokens=False,
                          return_offsets_mapping=True)["offset_mapping"]
    step = size - int(size * overlap_ratio)
    chunks = []
    for index, start in enumerate(range(0, len(offsets), step)):
        window = offsets[start:start + size]
        first, last = window[0][0], window[-1][1]
        pages = tuple(range(doc.page_at(first), doc.page_at(last - 1) + 1))
        chunks.append(Chunk(doc.name, index, doc.text[first:last], pages, len(window)))
        if start + size >= len(offsets):
            break  # the last window already reached the end of the document
    return chunks


def chunk_corpus(docs: list[Document], size: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, size)]
