"""Turn per-question results into summary tables and charts. Local only, no API calls.

    python make_report.py            # reads results/
    python make_report.py --mock     # reads results/mock/
"""
import argparse

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from raglab_path import add_src  # noqa: E402

add_src()

from raglab.config import CHUNK_SIZES, EMBEDDERS, RESULTS_DIR, TOP_KS  # noqa: E402
from raglab.metrics import bootstrap_ci, paired_diff  # noqa: E402

# Validated categorical slots 1 and 2; identity is also carried by marker shape and labels.
COLORS = {"minilm": "#2a78d6", "granite": "#eb6834"}
MARKERS = {3: "o", 5: "s"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlecolor": INK, "axes.titleweight": "bold",
})


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration."""
    rows = []
    for label, group in df.groupby("config", sort=False):
        answerable = group[group.answerable]
        unanswerable = group[~group.answerable]
        hits = answerable.hit.astype(float)
        scores = answerable.score / 2
        hit_low, hit_high = bootstrap_ci(hits)
        score_low, score_high = bootstrap_ci(scores)
        rows.append({
            "config": label, "chunk_size": group.chunk_size.iloc[0],
            "embedder": group.embedder.iloc[0], "top_k": group.top_k.iloc[0],
            "hit_rate": hits.mean(), "hit_low": hit_low, "hit_high": hit_high,
            "answer_score": scores.mean(), "score_low": score_low, "score_high": score_high,
            "fully_correct": (answerable.score == 2).mean(),
            "false_refusals": int(answerable.refused.sum()),
            "correct_declines": int((unanswerable.score == 2).sum()),
            "context_tokens": group.context_tokens.mean(),
            "retrieval_ms_p50": group.retrieval_ms.median(),
            "retrieval_ms_p95": group.retrieval_ms.quantile(0.95),
            "generation_s_p50": group.generation_s.median(),
            "generation_s_p95": group.generation_s.quantile(0.95),
            "unread_share": group.unread_share.iloc[0],
        })
    return pd.DataFrame(rows)


def factor_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compare levels of one factor, averaging the other factors within each question."""
    answerable = df[df.answerable].assign(hit=lambda d: d.hit.astype(float),
                                          score01=lambda d: d.score / 2)
    comparisons = [("chunk_size", 512, 256), ("chunk_size", 1024, 512), ("chunk_size", 1024, 256),
                   ("embedder", "granite", "minilm"), ("top_k", 5, 3)]
    rows = []
    for factor, a, b in comparisons:
        for metric in ("hit", "score01"):
            means = answerable.groupby([factor, "question"])[metric].mean()
            result = paired_diff(means[a].sort_index(), means[b].sort_index())
            rows.append({"comparison": f"{factor}: {a} vs {b}", "metric": metric, **result})
    return pd.DataFrame(rows)


def by_question_type(df: pd.DataFrame) -> pd.DataFrame:
    answerable = df[df.answerable].assign(hit=lambda d: d.hit.astype(float),
                                          score01=lambda d: d.score / 2)
    return answerable.groupby(["type", "chunk_size"])[["hit", "score01"]].mean().reset_index()


def per_question_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Hit rate and score per question across all 12 configurations, hardest first."""
    answerable = df[df.answerable].assign(hit=lambda d: d.hit.astype(float),
                                          score01=lambda d: d.score / 2)
    grid = answerable.groupby(["question", "type"])[["hit", "score01"]].mean().reset_index()
    return grid.sort_values("score01")


def judge_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare the judge with deterministic fact matching on distinct judged answers."""
    judged = df[df.answerable & (df.judged_by != "rule")]
    judged = judged.drop_duplicates(["question", "answer"]).copy()
    judged["judge_full"] = judged.score == 2
    judged["facts_full"] = judged.fact_share == 1
    disagree = judged[judged.judge_full != judged.facts_full]
    stats = {"judged_answers": len(judged), "agreement": float(1 - len(disagree) / max(len(judged), 1)),
             "disagreements": len(disagree)}
    columns = ["question", "score", "fact_share", "judge_reason", "answer"]
    return disagree[columns], stats


def plot_quality(summary: pd.DataFrame, path) -> None:
    """Small multiples: hit rate and answer score by chunk size, one panel per top-k."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, sharey=True)
    metrics = [("hit_rate", "hit_low", "hit_high", "Retrieval hit rate"),
               ("answer_score", "score_low", "score_high", "Answer score")]
    offsets = {"minilm": -0.05, "granite": 0.05}
    for row, (value, low, high, title) in enumerate(metrics):
        for col, k in enumerate(TOP_KS):
            ax = axes[row][col]
            for key in EMBEDDERS:
                part = summary[(summary.embedder == key) & (summary.top_k == k)]
                part = part.sort_values("chunk_size")
                x = [CHUNK_SIZES.index(c) + offsets[key] for c in part.chunk_size]
                ax.errorbar(x, part[value], color=COLORS[key], linewidth=2, marker=MARKERS[k],
                            markersize=7, capsize=0, elinewidth=1,
                            yerr=[part[value] - part[low], part[high] - part[value]], label=key)
            ax.set_title(f"{title}, top-{k}", loc="left", fontsize=10)
            ax.set_xticks(range(len(CHUNK_SIZES)), [str(c) for c in CHUNK_SIZES])
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Chunk size (tokens)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Quality by configuration (vertical lines: 95% bootstrap intervals, 16 questions)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cost_quality(summary: pd.DataFrame, path) -> None:
    """Answer score against context tokens: the trade-off a team actually pays for."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for key in EMBEDDERS:
        for k in TOP_KS:
            part = summary[(summary.embedder == key) & (summary.top_k == k)]
            ax.scatter(part.context_tokens, part.answer_score, s=70, color=COLORS[key],
                       marker=MARKERS[k], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=f"{key}, top-{k}")
    for size in CHUNK_SIZES:
        for k in TOP_KS:
            tokens = size * k
            ax.axvline(tokens, color=GRID, linewidth=0.8, zorder=1)
            ax.text(tokens, 1.04, f"{size}×{k}", ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([1000, 2000, 3000, 5000], ["1,000", "2,000", "3,000", "5,000"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Context tokens sent to the LLM per question (log scale)")
    ax.set_ylabel("Answer score")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.set_title("Answer score against context size", loc="left", fontsize=11, pad=18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_latency(summary: pd.DataFrame, path) -> None:
    """Two separate panels, because retrieval (ms) and generation (s) have different scales."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    order = list(CHUNK_SIZES)
    width = 0.36
    for ax, column, title, unit in (
            (axes[0], "retrieval_ms_p50", "Retrieval latency, median", "ms"),
            (axes[1], "generation_s_p50", "Generation latency, median", "s")):
        averaged = summary.groupby(["chunk_size", "embedder"])[column].mean()
        for i, key in enumerate(EMBEDDERS):
            values = [averaged[size, key] for size in order]
            x = [j + (i - 0.5) * (width + 0.02) for j in range(len(order))]
            bars = ax.bar(x, values, width, color=COLORS[key], label=key)
            ax.bar_label(bars, fmt="%.2f" if unit == "s" else "%.0f", fontsize=8, color=MUTED,
                         padding=2)
        ax.set_xticks(range(len(order)), [str(c) for c in order])
        ax.set_xlabel("Chunk size (tokens)")
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_table(summary: pd.DataFrame) -> str:
    lines = ["| Chunk | Embedder | k | Hit rate | Answer score | Fully correct | False refusals "
             "| Correct declines | Context tokens | Retrieval p50 | Generation p50 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    ordered = summary.sort_values(["answer_score", "hit_rate"], ascending=False)
    for r in ordered.itertuples():
        lines.append(
            f"| {r.chunk_size} | {r.embedder} | {r.top_k} | {r.hit_rate:.0%} | "
            f"{r.answer_score:.0%} ({r.score_low:.0%}–{r.score_high:.0%}) | {r.fully_correct:.0%} | "
            f"{r.false_refusals}/16 | {r.correct_declines}/4 | {r.context_tokens:,.0f} | "
            f"{r.retrieval_ms_p50:.0f} ms | {r.generation_s_p50:.2f} s |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    folder = RESULTS_DIR / "mock" if args.mock else RESULTS_DIR

    df = pd.read_csv(folder / "per_question.csv")
    df["answer"] = df["answer"].fillna("")
    summary = summarize(df)
    effects = factor_effects(df)
    types = by_question_type(df)
    grid = per_question_grid(df)
    disagreements, audit = judge_audit(df)

    summary.to_csv(folder / "summary.csv", index=False)
    effects.to_csv(folder / "factor_effects.csv", index=False)
    types.to_csv(folder / "by_question_type.csv", index=False)
    grid.to_csv(folder / "by_question.csv", index=False)
    disagreements.to_csv(folder / "judge_disagreements.csv", index=False)
    plot_quality(summary, folder / "quality_by_config.png")
    plot_cost_quality(summary, folder / "score_vs_context.png")
    plot_latency(summary, folder / "latency.png")

    table = markdown_table(summary)
    (folder / "results_table.md").write_text(table + "\n", encoding="utf-8")
    pd.set_option("display.width", 160, "display.max_columns", 30, "display.max_colwidth", 70)
    print(table, "\n\nFACTOR EFFECTS\n", effects.round(3).to_string(index=False))
    print("\nBY TYPE\n", types.round(3).to_string(index=False))
    print("\nBY QUESTION\n", grid.round(3).to_string(index=False))
    print("\nJUDGE AUDIT", audit)


if __name__ == "__main__":
    main()
