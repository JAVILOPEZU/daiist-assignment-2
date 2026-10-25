"""
Turn per-mushroom poison scores into decisions and euros.

Shared by train.py (to sweep and pick the operating point) and app.py (to
recompute everything live when the sliders move), so both always agree.

Every ground-truth mushroom gets one poison score s in [0, 1] (see
train.poison_score). Two thresholds split it into three outcomes:

    s <  t_low             -> PASS   (model says edible, it gets eaten)
    t_low <= s < t_high    -> REVIEW (a person checks it)
    s >= t_high            -> FAIL   (model says poisonous, it is thrown away)

Positive class = Poisonous. So a poisonous mushroom that passes is a false
negative, and an edible one that fails is a false positive. The single
threshold case is just t_low == t_high.
"""

import json
from pathlib import Path

import numpy as np

COSTS_FILE = Path(__file__).resolve().parent / "costs.json"
COST_KEYS = ("fn", "fp", "tp", "tn", "review", "base_rate")


def load_costs(path: Path = COSTS_FILE) -> dict:
    """Read the euro figures and base rate from costs.json."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in COST_KEYS if k not in raw]
    if missing:
        raise KeyError(f"{path.name} is missing: {', '.join(missing)}")
    return {k: float(raw[k]) for k in COST_KEYS}


def outcome_rates(scores, poisonous, t_low: float, t_high: float) -> dict:
    """Share of each class that passes, goes to review, or fails."""
    scores = np.asarray(scores, dtype=float)
    poisonous = np.asarray(poisonous, dtype=bool)
    passed = scores < t_low
    failed = scores >= t_high
    review = ~passed & ~failed

    def share(mask, group):
        return float(mask[group].mean()) if group.any() else 0.0

    pos, neg = poisonous, ~poisonous
    return {
        "pos_pass": share(passed, pos),    # FN: poisonous mushroom eaten
        "pos_review": share(review, pos),
        "pos_fail": share(failed, pos),    # TP
        "neg_pass": share(passed, neg),    # TN
        "neg_review": share(review, neg),
        "neg_fail": share(failed, neg),    # FP: edible mushroom thrown away
    }


def expected_cost(scores, poisonous, t_low: float, t_high: float, costs: dict) -> dict:
    """Expected euros and review share per 1,000 mushrooms at the given base rate.

    Rates are measured within each class and then reweighted by the assumed
    base rate, so the answer does not depend on the class mix of the split.
    A reviewed mushroom costs the review fee and is assumed to be judged
    correctly by the person.
    """
    r = outcome_rates(scores, poisonous, t_low, t_high)
    pi = costs["base_rate"]
    per_case = (
        pi * (r["pos_pass"] * costs["fn"] + r["pos_fail"] * costs["tp"]
              + r["pos_review"] * costs["review"])
        + (1 - pi) * (r["neg_pass"] * costs["tn"] + r["neg_fail"] * costs["fp"]
                      + r["neg_review"] * costs["review"])
    )
    review_share = pi * r["pos_review"] + (1 - pi) * r["neg_review"]
    return {**r, "cost_per_1000": 1000 * per_case, "review_share": review_share}


def confusion_counts(scores, poisonous, t_low: float, t_high: float) -> dict:
    """Raw counts on the split itself (not reweighted), for the dashboard."""
    scores = np.asarray(scores, dtype=float)
    poisonous = np.asarray(poisonous, dtype=bool)
    passed = scores < t_low
    failed = scores >= t_high
    review = ~passed & ~failed
    return {
        "tp": int((failed & poisonous).sum()),
        "fn": int((passed & poisonous).sum()),
        "fp": int((failed & ~poisonous).sum()),
        "tn": int((passed & ~poisonous).sum()),
        "review_pos": int((review & poisonous).sum()),
        "review_neg": int((review & ~poisonous).sum()),
    }


def sweep(scores, poisonous, costs: dict, thresholds) -> list[dict]:
    """Single-threshold sweep: cost at every t with no review band."""
    return [
        {"threshold": float(t), **expected_cost(scores, poisonous, t, t, costs)}
        for t in thresholds
    ]


def best_band(scores, poisonous, costs: dict, grid) -> dict:
    """Cheapest (t_low, t_high) pair on the grid, with t_low <= t_high."""
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            result = expected_cost(scores, poisonous, lo, hi, costs)
            if best is None or result["cost_per_1000"] < best["cost_per_1000"]:
                best = {"t_low": float(lo), "t_high": float(hi), **result}
    return best


def policy_costs(scores, poisonous, costs: dict, t_single: float,
                 t_low: float, t_high: float) -> dict:
    """The design priced against the three alternatives the brief asks for."""
    return {
        "do_nothing": expected_cost(scores, poisonous, np.inf, np.inf, costs),
        "person_checks_all": expected_cost(scores, poisonous, -np.inf, np.inf, costs),
        "model_alone": expected_cost(scores, poisonous, t_single, t_single, costs),
        "model_plus_review": expected_cost(scores, poisonous, t_low, t_high, costs),
    }
