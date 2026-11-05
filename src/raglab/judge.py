"""Answer scoring: the LLM judge, plus a deterministic fact check used to audit the judge."""
import hashlib
import json
import re
import unicodedata

from .config import RUBRIC_PATH
from .llm import LLM
from .prompts import REFUSAL, judge_prompt


def is_refusal(answer: str) -> bool:
    return REFUSAL.lower().rstrip(".") in answer.lower()


def _clean(text: str) -> str:
    """Flatten LaTeX and Unicode so '$10^{-8}$' and '10−8' both compare as '10-8'."""
    text = unicodedata.normalize("NFKC", text).lower().replace("−", "-")
    text = re.sub(r"[$\\{}^*]", "", text)
    return re.sub(r"\s+", " ", text)


def fact_match(answer: str, facts: list) -> float:
    """Share of expected facts present. A fact given as a list accepts any alternative."""
    if not facts:
        return 0.0
    text = _clean(answer)
    found = [any(_clean(alt) in text for alt in (f if isinstance(f, list) else [f])) for f in facts]
    return sum(found) / len(found)


def score_question(llm: LLM, question: dict, answers: list[str]) -> dict[str, dict]:
    """Return {answer text: {"score", "reason", "by"}} for one question's answers.

    Refusals are scored by rule, because the correct score is known without a judge. The
    remaining distinct answers go to the judge in one call. They carry neutral ids in a
    hash-based order, so the judge never learns which configuration produced an answer.
    """
    results, to_judge = {}, []
    for answer in sorted(set(answers), key=lambda a: hashlib.sha256(a.encode()).hexdigest()):
        if is_refusal(answer):
            score = 0 if question["answerable"] else 2
            results[answer] = {"score": score, "reason": "exact refusal", "by": "rule"}
        else:
            to_judge.append(answer)
    if not to_judge:
        return results

    if llm.provider == "mock":  # offline stand-in for the judge: no API call
        for answer in to_judge:
            share = fact_match(answer, question["facts"])
            score = 2 if share == 1 else 1 if share > 0 else 0
            results[answer] = {"score": score, "reason": "mock fact match", "by": "mock"}
        return results

    candidates = {f"A{i}": answer for i, answer in enumerate(to_judge, 1)}
    prompt = judge_prompt(RUBRIC_PATH.read_text(encoding="utf-8"), question["question"],
                          question["reference"], candidates)
    verdicts = json.loads(llm.generate(prompt, json_mode=True).text)
    for candidate_id, answer in candidates.items():
        verdict = verdicts[candidate_id]  # a missing id fails loudly rather than scoring 0
        if verdict["score"] not in (0, 1, 2):
            raise ValueError(f"{question['id']}: invalid score {verdict['score']}")
        results[answer] = {"score": verdict["score"], "reason": verdict["reason"], "by": "judge"}
    return results
