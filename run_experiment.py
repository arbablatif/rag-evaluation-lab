"""Run the RAG grid: retrieval (local), then generation and judging (API, cached).

Examples:
    python run_experiment.py --mock                 # end to end, no API calls
    python run_experiment.py --stage retrieval      # local only, prints the API call plan
    python run_experiment.py --stage generate --pilot   # one configuration only
    python run_experiment.py --stage all            # everything, resumes from the cache
"""
import argparse
import json
import random
import statistics
import time
from collections import defaultdict

import pandas as pd

from raglab_path import add_src  # noqa: F401  (puts src/ on sys.path)

add_src()

from raglab.chunker import Chunk, chunk_corpus  # noqa: E402
from raglab.config import (CHUNK_SIZES, EMBEDDERS, PAPERS_DIR, QUESTIONS_PATH,  # noqa: E402
                           RESULTS_DIR, Config, all_configs)
from raglab.embedder import embed_chunks, embed_query, unread_share  # noqa: E402
from raglab.judge import fact_match, is_refusal, score_question  # noqa: E402
from raglab.llm import LLM, BudgetExceeded, QuotaExhausted  # noqa: E402
from raglab.loader import load_corpus  # noqa: E402
from raglab.prompts import answer_prompt  # noqa: E402
from raglab.retriever import is_hit, top_k  # noqa: E402

PILOT = Config(512, "granite", 5)
LATENCY_ROUNDS = 7  # each query is timed once per round; the median is reported


def build_indexes(docs):
    """Chunk once per size and embed once per (model, size)."""
    chunks = {size: chunk_corpus(docs, size) for size in CHUNK_SIZES}
    indexes = {}
    for size in CHUNK_SIZES:
        for key in EMBEDDERS:
            vectors, meta = embed_chunks(key, size, chunks[size])
            meta["unread_share"] = unread_share(key, chunks[size])
            indexes[key, size] = vectors, meta
            print(f"  index {key}-{size}: {meta['n_chunks']} chunks, "
                  f"{meta['unread_share']:.0%} of chunk text unread, {meta['index_seconds']}s")
    return chunks, indexes


def retrieve_all(questions, configs, chunks, indexes) -> tuple[list[dict], dict]:
    """Run retrieval for every configuration. Returns rows and the retrieved chunks."""
    timings = time_retrieval(questions, configs, indexes)
    rows, retrieved = [], {}
    for config in configs:
        vectors, meta = indexes[config.embedder, config.chunk_size]
        pool: list[Chunk] = chunks[config.chunk_size]
        for q in questions:
            ids = top_k(embed_query(config.embedder, q["question"]), vectors, config.top_k)
            hits = [pool[i] for i in ids]
            retrieved[config.label, q["id"]] = hits
            rows.append({
                "config": config.label, "chunk_size": config.chunk_size,
                "embedder": config.embedder, "top_k": config.top_k,
                "question": q["id"], "type": q["type"], "answerable": q["answerable"],
                "hit": is_hit(hits, q["evidence"]) if q["answerable"] else None,
                "context_tokens": sum(c.n_tokens for c in hits),
                "retrieval_ms": statistics.median(timings[config.label, q["id"]]),
                "unread_share": meta["unread_share"],
                "index_seconds": meta["index_seconds"],
            })
    return rows, retrieved


def time_retrieval(questions, configs, indexes) -> dict:
    """Time query embedding plus search in shuffled rounds.

    Timing one configuration after another lets a burst of background CPU load land on
    whichever configuration is running. Interleaving spreads that noise across all of them.
    """
    for key in EMBEDDERS:  # keep model start-up and CPU warm-up out of the timing
        for q in questions:
            embed_query(key, q["question"])
    rng = random.Random(7)
    timings = defaultdict(list)
    for _ in range(LATENCY_ROUNDS):
        order = list(configs)
        rng.shuffle(order)
        for config in order:
            vectors = indexes[config.embedder, config.chunk_size][0]
            for q in questions:
                start = time.perf_counter()
                top_k(embed_query(config.embedder, q["question"]), vectors, config.top_k)
                timings[config.label, q["id"]].append((time.perf_counter() - start) * 1000)
    return timings


def plan_calls(llm: LLM, questions, configs, retrieved) -> dict:
    """Count generation calls before spending anything."""
    prompts = {answer_prompt(q["question"], retrieved[c.label, q["id"]])
               for c in configs for q in questions}
    new = sum(not llm.is_cached(p) for p in prompts)
    return {"pairs": len(configs) * len(questions), "unique_prompts": len(prompts), "new": new}


def generate_all(llm: LLM, questions, configs, retrieved, rows) -> list[dict]:
    """Send each prompt once. Duplicates come back from the cache at no cost."""
    by_key = {(r["config"], r["question"]): r for r in rows}
    answered = []
    for config in configs:
        for q in questions:
            row = dict(by_key[config.label, q["id"]])
            reply = llm.generate(answer_prompt(q["question"], retrieved[config.label, q["id"]]))
            row.update(answer=reply.text, generation_s=reply.latency_s,
                       input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                       refused=is_refusal(reply.text))
            answered.append(row)
        print(f"  generated {config.label} (API attempts so far: {llm.calls_made()})")
    return answered


def judge_all(llm: LLM, questions, answered) -> list[dict]:
    """One judge call per question covers every configuration's answer."""
    for q in questions:
        rows = [r for r in answered if r["question"] == q["id"]]
        verdicts = score_question(llm, q, [r["answer"] for r in rows])
        for row in rows:
            row.update(score=verdicts[row["answer"]]["score"],
                       judge_reason=verdicts[row["answer"]]["reason"],
                       judged_by=verdicts[row["answer"]]["by"],
                       fact_share=fact_match(row["answer"], q["facts"]))
        print(f"  judged {q['id']} (API attempts so far: {llm.calls_made()})")
    return answered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="no API calls; fake LLM and judge")
    parser.add_argument("--stage", choices=["retrieval", "generate", "all"], default="all")
    parser.add_argument("--pilot", action="store_true", help="generate for one configuration")
    args = parser.parse_args()

    out = RESULTS_DIR / "mock" if args.mock else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    configs = all_configs()
    llm = LLM("mock" if args.mock else None)

    print("Building indexes (local)...")
    docs = load_corpus(PAPERS_DIR)
    chunks, indexes = build_indexes(docs)

    print("Retrieval (local)...")
    rows, retrieved = retrieve_all(questions, configs, chunks, indexes)
    pd.DataFrame(rows).to_csv(out / "retrieval.csv", index=False)

    run_configs = [PILOT] if args.pilot else configs
    plan = plan_calls(llm, questions, run_configs, retrieved)
    print(f"Generation plan: {plan['pairs']} config-question pairs, "
          f"{plan['unique_prompts']} unique prompts, {plan['new']} not yet cached.")
    if args.stage == "retrieval":
        return

    try:
        print(f"Generation with {llm.provider}/{llm.model}...")
        answered = generate_all(llm, questions, run_configs, retrieved, rows)
        if args.stage == "generate":
            pd.DataFrame(answered).to_csv(out / "generated.csv", index=False)
            return
        print("Judging...")
        scored = judge_all(llm, questions, answered)
    except (BudgetExceeded, QuotaExhausted) as err:
        raise SystemExit(f"Stopped safely, finished work is cached: {err}")

    pd.DataFrame(scored).to_csv(out / "per_question.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "provider": llm.provider, "model": llm.model, "api_attempts": llm.calls_made(),
        "unique_generation_prompts": plan["unique_prompts"],
    }, indent=2))
    print(f"Done. Results in {out}. Total API attempts: {llm.calls_made()}")


if __name__ == "__main__":
    main()
