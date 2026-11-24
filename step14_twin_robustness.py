"""Does risk-ranked dispatch beat first come, first served across assumptions, or only at the app's defaults?

Runs the shift simulator over a grid of unit counts, response times and patience limits, with many random shifts per
setting. Each shift is simulated under both policies (paired), and the difference in the share of serious/fatal calls
that received an enhanced unit is summarised with a bootstrap interval. Also sweeps the age bonus that stops low-risk
calls waiting forever. Output: reports/twin_robustness.csv, reports/twin_robustness.json
"""
import itertools
import json

import numpy as np
import pandas as pd

import shift_sim
from common import ROOT

UNITS, SERVICE, PATIENCE = (6, 12, 24), (20, 30, 45), (10, 20, 40)
SEEDS = range(20)
N_CALLS = 600
AGE_WEIGHTS = (0.0, 0.002, 0.01, 0.05)


def paired_diff(replay, units, service, patience, age_weight=0.002):
    diffs = []
    for seed in SEEDS:
        shift = shift_sim.build_shift(replay, N_CALLS, 480, seed)
        share = {p: shift_sim.summarize(shift_sim.simulate(shift, units, service, patience, p, age_weight=age_weight))["serious_caught_share"]
                 for p in shift_sim.POLICIES}
        diffs.append(share["model"] - share["fifo"])
    return np.array(diffs)


def interval(diffs, rng):
    boots = [rng.choice(diffs, len(diffs)).mean() for _ in range(2000)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    rng = np.random.default_rng(0)
    replay = pd.read_csv(ROOT / "data/processed/replay_2025.csv", usecols=["score", "target"])
    rows = []
    for units, service, patience in itertools.product(UNITS, SERVICE, PATIENCE):
        d = paired_diff(replay, units, service, patience)
        lo, hi = interval(d, rng)
        capacity = units * 480 / service / N_CALLS
        rows.append(dict(units=units, service_min=service, patience_min=patience, capacity_share_of_calls=round(capacity, 3),
                         mean_gain_points=100 * d.mean(), ci_lo=100 * lo, ci_hi=100 * hi, wins=int((d > 0).sum()), shifts=len(d)))
        print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(ROOT / "reports/twin_robustness.csv", index=False)

    ages = []
    for w in AGE_WEIGHTS:
        d = paired_diff(replay, 12, 30, 20, age_weight=w)
        lo, hi = interval(d, rng)
        ages.append(dict(age_weight=w, mean_gain_points=100 * d.mean(), ci_lo=100 * lo, ci_hi=100 * hi))

    scarce = grid[grid.capacity_share_of_calls < 0.5]
    summary = {
        "settings": len(grid), "shifts_per_setting": len(SEEDS), "calls_per_shift": N_CALLS,
        "settings_where_model_clearly_better": int((grid.ci_lo > 0).sum()),
        "settings_where_model_clearly_worse": int((grid.ci_hi < 0).sum()),
        "settings_inconclusive": int(((grid.ci_lo <= 0) & (grid.ci_hi >= 0)).sum()),
        "median_gain_points_all": float(grid.mean_gain_points.median()),
        "median_gain_points_when_units_cover_under_half_of_calls": float(scarce.mean_gain_points.median()) if len(scarce) else None,
        "best_setting": grid.loc[grid.mean_gain_points.idxmax()].to_dict(),
        "worst_setting": grid.loc[grid.mean_gain_points.idxmin()].to_dict(),
        "age_weight_sweep_units12_service30_patience20": ages,
    }
    (ROOT / "reports/twin_robustness.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
