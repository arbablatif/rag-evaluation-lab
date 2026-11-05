"""Fast unit tests. None of these touch the network or load an embedding model."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.chunker import Chunk, chunk_document, tokenizer  # noqa: E402
from raglab.config import QUESTIONS_PATH  # noqa: E402
from raglab.judge import fact_match, is_refusal, score_question  # noqa: E402
from raglab.llm import LLM, BudgetExceeded  # noqa: E402
from raglab.loader import Document, normalize  # noqa: E402
from raglab.metrics import bootstrap_ci, paired_diff  # noqa: E402
from raglab.prompts import REFUSAL, answer_prompt  # noqa: E402
from raglab.retriever import is_hit, top_k  # noqa: E402


def make_chunk(text, index=0, doc="a.pdf"):
    return Chunk(doc, index, text, (1,), 10)


def test_chunks_respect_size_overlap_and_pages():
    page = "word " * 300
    doc = Document("d.pdf", page + "\n" + page, [0, len(page) + 1])
    chunks = chunk_document(doc, 256)
    assert all(c.n_tokens <= 256 for c in chunks)
    assert any(len(c.pages) == 2 for c in chunks)  # windows do cross the page break
    tokens = sum(len(tokenizer()(c.text, add_special_tokens=False)["input_ids"]) for c in chunks)
    assert tokens > 600  # overlap means more tokens in chunks than in the document


def test_normalize_handles_ligatures_and_hyphenation():
    assert normalize("efﬁcient opti-\nmizer") == "efficient optimizer"


def test_hit_needs_every_evidence_group():
    chunks = [make_chunk("alpha beta"), make_chunk("gamma")]
    assert is_hit(chunks, [["alpha"], ["gamma", "missing"]])
    assert not is_hit(chunks, [["alpha"], ["delta"]])


def test_top_k_orders_by_similarity():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    assert top_k(np.array([1.0, 0.0]), matrix, 2) == [0, 2]


def test_same_chunks_give_same_prompt_in_any_order():
    a, b = make_chunk("first", 0), make_chunk("second", 1)
    assert answer_prompt("q?", [a, b]) == answer_prompt("q?", [b, a])


def test_fact_match_handles_latex_and_alternatives():
    assert fact_match("epsilon is $10^{-8}$", [["10-8", "1e-8"]]) == 1
    assert fact_match("only 8 heads", ["8", "64"]) == 0.5


def test_refusals_are_scored_by_rule_without_a_judge():
    llm = LLM("mock")
    answerable = {"answerable": True, "facts": ["x"]}
    unanswerable = {"answerable": False, "facts": []}
    assert is_refusal(REFUSAL)
    assert score_question(llm, answerable, [REFUSAL])[REFUSAL]["score"] == 0
    assert score_question(llm, unanswerable, [REFUSAL])[REFUSAL]["score"] == 2


def test_cache_prevents_a_second_call_and_cap_stops_a_new_one(tmp_path, monkeypatch):
    monkeypatch.setattr("raglab.llm.MIN_CALL_INTERVAL_S", 0)
    llm = LLM("gemini", cache_dir=tmp_path, max_calls=1)
    calls = []

    def fake(prompt, json_mode):
        calls.append(prompt)
        return "answer", 3, 1

    monkeypatch.setattr(llm, "_gemini", fake)
    assert llm.generate("p1").cached is False
    assert llm.generate("p1").cached is True
    assert len(calls) == 1
    with pytest.raises(BudgetExceeded):
        llm.generate("p2")
    assert len(calls) == 1


def test_paired_diff_separates_only_clear_differences():
    assert paired_diff([1] * 20, [0] * 20)["separated"] is True
    assert paired_diff([1, 0] * 10, [0, 1] * 10)["separated"] is False
    low, high = bootstrap_ci([0, 1] * 10)
    assert low < 0.5 < high


def test_question_set_shape():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    assert len(questions) == 20 and len({q["id"] for q in questions}) == 20
    assert sum(q["answerable"] for q in questions) == 16
    assert all(bool(q["evidence"]) == q["answerable"] for q in questions)
