"""Uncertainty estimates. With about 20 questions, a score means little without an interval."""
import numpy as np

N_BOOTSTRAP = 5000
SEED = 7


def bootstrap_ci(values, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean, resampling questions with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[picks].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def paired_diff(a, b) -> dict:
    """Mean of (a - b) over the same questions, with a bootstrap interval.

    Pairing removes question difficulty from the comparison. If the interval includes 0,
    the data cannot separate the two options.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = bootstrap_ci(diff)
    return {"diff": float(diff.mean()), "ci_low": low, "ci_high": high,
            "separated": bool(low > 0 or high < 0)}
