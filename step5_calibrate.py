"""Calibration, uncertainty bands, replay data and operating curves for model v2.

Out-of-time scores: for every year Y from 2022, train on all earlier years and score Y. That gives honest scores
for 2022..2025. The isotonic calibrator is fitted on 2022..2024 and checked once on 2025.
Uncertainty is empirical: for each calibrated-score bin, the Wilson interval of the serious/fatal rate actually
observed among similar historic calls.
Outputs: models/calibrator_v2.joblib, reports/calibration_v2.csv, reports/capacity_curve.csv,
data/processed/replay_2025.csv (2025 out-of-time scores with coordinates, used by the control room).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from common import ROOT, TIERS
from step4_train_multiyear import fit_lgbm, load_clean, score_lgbm

TIER = "T4_plus_force"
CAL_PATH = ROOT / "models/calibrator_v2.joblib"
REPLAY_PATH = ROOT / "data/processed/replay_2025.csv"
N_BINS = 12


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def interval_table(y, p, n_bins=N_BINS):
    """Equal-count bins of the calibrated score with the observed rate and its Wilson 95% interval."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        rows.append(dict(score_low=float(edges[b]), score_high=float(edges[b + 1]), n=n, observed_rate=k / n, ci_low=lo, ci_high=hi))
    rows[0]["score_low"], rows[-1]["score_high"] = 0.0, 1.0
    return pd.DataFrame(rows)


def main():
    df, categories = load_clean()
    features = TIERS[TIER]
    parts = []
    for year in [y for y in sorted(df.year.unique()) if y >= 2022]:
        train, test = df[df.year < year], df[df.year == year].copy()
        model = fit_lgbm(train, features, categories)
        test["raw_score"] = score_lgbm(model, test, features, categories)
        parts.append(test)
        print(f"scored {year}: n={len(test):,}")
    oot = pd.concat(parts, ignore_index=True)

    fit = oot[oot.year <= 2024]
    holdout = oot[oot.year == 2025]
    check = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(fit.raw_score, fit.target)
    cal_hold = check.predict(holdout.raw_score)
    y = holdout.target.to_numpy()
    report = dict(ece_raw=ece(y, holdout.raw_score.to_numpy()), ece_calibrated=ece(y, cal_hold),
                  brier_raw=brier_score_loss(y, holdout.raw_score), brier_calibrated=brier_score_loss(y, cal_hold),
                  brier_constant=brier_score_loss(y, np.full(len(y), fit.target.mean())))
    print({k: round(v, 4) for k, v in report.items()})

    final = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oot.raw_score, oot.target)
    table = interval_table(oot.target.to_numpy(), final.predict(oot.raw_score))
    joblib.dump({"isotonic": final, "intervals": table, "holdout_report": report, "n_fit": len(oot)}, CAL_PATH)
    table.to_csv(ROOT / "reports/calibration_v2.csv", index=False)

    # 2025 replay file + precision-vs-K curve (both out-of-time).
    holdout = holdout.copy()
    holdout["score"] = cal_hold
    keep = ["year", "latitude", "longitude", "road_type", "speed_limit", "weather", "light_condition", "junction_detail",
            "hour", "day_of_week", "month", "urban_or_rural_area", "first_road_class", "junction_control",
            "road_surface_conditions", "police_force", "target", "raw_score", "score"]
    holdout[keep].to_csv(REPLAY_PATH, index=False)
    order = np.argsort(-holdout.raw_score.to_numpy(), kind="stable")
    sorted_y = holdout.target.to_numpy()[order]
    cum = np.cumsum(sorted_y)
    ks = np.unique(np.concatenate([np.arange(50, 2000, 50), np.arange(2000, len(sorted_y), 1000)]))
    curve = pd.DataFrame(dict(k=ks, caught=cum[ks - 1], precision=cum[ks - 1] / ks))
    curve["share_of_all_serious"] = curve.caught / sorted_y.sum()
    curve["flag_share"] = curve.k / len(sorted_y)
    curve.to_csv(ROOT / "reports/capacity_curve.csv", index=False)
    pd.Series(report).to_json(ROOT / "reports/calibration_report.json", indent=2)
    print(table.round(3).to_string(index=False))
    print("Saved", CAL_PATH.name, REPLAY_PATH.name)


if __name__ == "__main__":
    main()
