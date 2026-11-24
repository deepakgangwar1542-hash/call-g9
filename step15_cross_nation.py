"""Does a model trained only on England still rank risk in Wales and Scotland?

Train on English collisions from 2021-2024; test on 2025 collisions in England (same-nation control), Wales and
Scotland. Police-force codes are nation-specific, so only tiers without police_force are used. Reports ranking quality
(AUC, precision lift at the 2% flag rate) and calibration (mean predicted risk against the actual serious/fatal rate).
Output: reports/cross_nation.csv, reports/cross_nation.json
"""
import json

import pandas as pd

from common import ROOT, TIERS
from step4_train_multiyear import fit_lgbm, level1_scores, load_clean, metrics, score_lgbm

WALES, SCOTLAND = {60, 61, 62, 63}, {99}
TIER_NAMES = ("T2_plus_time", "T3_plus_context")


def nation(force: pd.Series) -> pd.Series:
    return force.map(lambda f: "Scotland" if f in SCOTLAND else "Wales" if f in WALES else "England")


def main():
    df, categories = load_clean()
    df["nation"] = nation(df["police_force"])
    train = df[(df.nation == "England") & (df.year <= 2024)]
    test = df[df.year == 2025]
    rows = []
    for tier in TIER_NAMES:
        features = TIERS[tier]
        model = fit_lgbm(train, features, categories)
        for name, part in test.groupby("nation"):
            scores = score_lgbm(model, part, features, categories)
            baseline = level1_scores(train, part)
            rows.append({"tier": tier, "test_nation": name, **metrics(part.target, scores),
                         "mean_predicted": float(scores.mean()), "level1_P@2pct_lift": metrics(part.target, baseline)["P@2pct_lift"]})
    out = pd.DataFrame(rows)
    out["calibration_gap_points"] = 100 * (out.mean_predicted - out.base_rate)
    out.to_csv(ROOT / "reports/cross_nation.csv", index=False)
    cols = ["tier", "test_nation", "n", "base_rate", "roc_auc", "P@2pct_lift", "level1_P@2pct_lift", "mean_predicted", "calibration_gap_points"]
    print(out[cols].round(3).to_string(index=False))
    (ROOT / "reports/cross_nation.json").write_text(json.dumps(out[cols].round(4).to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
