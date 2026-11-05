"""Experiment grid, fixed settings and paths. Everything the experiment varies is here."""
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

PAPERS_DIR = ROOT / "data" / "papers"
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RUBRIC_PATH = ROOT / "eval" / "rubric.md"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"

# ---- The three factors under test ----
CHUNK_SIZES = (256, 512, 1024)  # tokens
EMBEDDERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "granite": "ibm-granite/granite-embedding-small-english-r2",
}
TOP_KS = (3, 5)

# ---- Held constant across all configurations ----
OVERLAP_RATIO = 0.15
# One tokenizer measures every chunk, so both embedders see identical chunks.
CHUNK_TOKENIZER = EMBEDDERS["minilm"]

# ---- LLM ----
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
    "mock": "mock",
}
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
MODEL_OVERRIDE = os.getenv("LLM_MODEL", "").strip()
MAX_API_CALLS = int(os.getenv("MAX_API_CALLS", "300"))  # hard cap, counted on disk
MIN_CALL_INTERVAL_S = float(os.getenv("MIN_CALL_INTERVAL_S", "4.5"))  # throttle


@dataclass(frozen=True)
class Config:
    chunk_size: int
    embedder: str
    top_k: int

    @property
    def label(self) -> str:
        return f"{self.chunk_size}-{self.embedder}-k{self.top_k}"


def all_configs() -> list[Config]:
    return [Config(c, e, k) for c, e, k in product(CHUNK_SIZES, EMBEDDERS, TOP_KS)]
