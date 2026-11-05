"""Extract PDF text page by page and join each paper into one continuous document."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    name: str
    text: str                # all pages joined with a single newline
    page_starts: list[int]   # character offset where each page begins (index 0 = page 1)

    def page_at(self, offset: int) -> int:
        """Return the 1-indexed page containing a character offset."""
        page = 1
        for i, start in enumerate(self.page_starts, 1):
            if start > offset:
                break
            page = i
        return page


def load_pdf(path: Path) -> Document:
    parts, starts, pos = [], [], 0
    for page in PdfReader(str(path)).pages:
        text = page.extract_text() or ""
        starts.append(pos)
        parts.append(text)
        pos += len(text) + 1  # +1 for the joining newline
    return Document(path.name, "\n".join(parts), starts)


def load_corpus(folder: Path) -> list[Document]:
    docs = [load_pdf(p) for p in sorted(folder.glob("*.pdf"))]
    if not docs:
        raise SystemExit(f"No PDFs in {folder}. Run: python scripts/download_papers.py")
    return docs


def normalize(text: str) -> str:
    """Canonical form for evidence matching: no ligatures or line-break hyphens, lowercase."""
    text = unicodedata.normalize("NFKC", text)  # e.g. the single-glyph "fi" ligature -> "fi"
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text).lower().strip()
