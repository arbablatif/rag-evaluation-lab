"""The answer prompt and the judge prompt. Both are fixed across all configurations."""
import json

from .chunker import Chunk

REFUSAL = "The provided documents do not contain this information."


def answer_prompt(question: str, chunks: list[Chunk]) -> str:
    """Build the answer prompt with passages in document order, not ranking order.

    Using a canonical order means two configurations that retrieve the same set of chunks
    produce the same prompt, so the cache sends it only once. It also holds passage order
    constant, which would otherwise be one more thing varying between configurations.
    """
    ordered = sorted(chunks, key=lambda c: (c.doc, c.index))
    passages = "\n\n".join(f"[{i}]\n{c.text}" for i, c in enumerate(ordered, 1))
    return (
        "Answer the question using only the passages below. Be concise and include every "
        "detail the question asks for. Do not use outside knowledge.\n"
        f'If the passages do not contain the answer, reply exactly: "{REFUSAL}"\n\n'
        f"{passages}\n\nQuestion: {question}\nAnswer:"
    )


def judge_prompt(rubric: str, question: str, reference: str, candidates: dict[str, str]) -> str:
    """One prompt grades every distinct answer to a question, which saves most judge calls."""
    return (
        f"{rubric}\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answers (JSON):\n{json.dumps(candidates, ensure_ascii=False, indent=1)}\n\n"
        "Return only a JSON object mapping every candidate id to "
        '{"score": 0, 1 or 2, "reason": "at most 12 words"}.'
    )
