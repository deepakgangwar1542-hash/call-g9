"""Model bake-off: LightGBM vs CatBoost vs XGBoost vs a rank-average ensemble.

Same temporal protocol as step4: train on 2021..2024, test once on 2025, same feature tier.
Differences are reported with a paired bootstrap interval against LightGBM, so a "win" must clear zero.
Writes reports/bakeoff.csv and reports/bakeoff.json.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

from common import FLAG_RATE, LGBM_PARAMS, MULTI_CATEGORICAL, ROOT, SEED, TIERS, precision_at_k
from step4_train_multiyear import load_clean, metrics, to_frame

REPORTS = ROOT / "reports"
TIER = "T4_plus_force"
BOOTSTRAPS = 300


def fit_lightgbm(train, features, categories):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(to_frame(train, features, categories), train["target"])
    return lambda frame: model.predict_proba(to_frame(frame, features, categories))[:, 1]


def fit_xgboost(train, features, categories):
    model = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=20, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                              random_state=SEED, n_jobs=-1, verbosity=0)
    model.fit(to_frame(train, features, categories), train["target"])
    return lambda frame: model.predict_proba(to_frame(frame, features, categories))[:, 1]


def fit_catboost(train, features, categories):
    cat_cols = [c for c in features if c in categories]

    def prep(frame):
        out = frame[features].copy()
        for c in cat_cols:
            out[c] = out[c].astype(str)
        return out

    model = CatBoostClassifier(iterations=600, learning_rate=0.08, depth=6, random_seed=SEED, verbose=0,
                               cat_features=cat_cols, thread_count=-1)
    model.fit(prep(train), train["target"])
    return lambda frame: model.predict_proba(prep(frame))[:, 1]


def rank_average(*score_sets):
    return np.mean([rankdata(s) / len(s) for s in score_sets], axis=0)


def paired_bootstrap(y, candidate, reference, rate=FLAG_RATE, n=BOOTSTRAPS):
    rng, y = np.random.default_rng(SEED), np.asarray(y)
    diffs = []
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        k = max(1, int(np.ceil(rate * len(idx))))
        diffs.append(precision_at_k(y[idx], candidate[idx], k, seed=SEED + i)
                     - precision_at_k(y[idx], reference[idx], k, seed=SEED + i))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def main():
    df, categories = load_clean()
    years = sorted(df.year.unique())
    test_year = years[-1]
    train, test = df[df.year < test_year], df[df.year == test_year]
    features = TIERS[TIER]
    print(f"Train {years[0]}-{years[-2]} ({len(train):,} rows), test {test_year} ({len(test):,} rows), {len(features)} features")

    scorers = {"LightGBM": fit_lightgbm, "XGBoost": fit_xgboost, "CatBoost": fit_catboost}
    scores = {}
    for name, fit in scorers.items():
        print("fitting", name, flush=True)
        scores[name] = fit(train, features, categories)(test)
    scores["Ensemble (rank average of three)"] = rank_average(*scores.values())

    rows = []
    for name, s in scores.items():
        row = {"model": name, **metrics(test.target, s)}
        if name != "LightGBM":
            d, lo, hi = paired_bootstrap(test.target, s, scores["LightGBM"])
            row.update({"P@2pct_diff_vs_lightgbm": d, "diff_ci_low": lo, "diff_ci_high": hi,
                        "ci_excludes_zero": bool(lo > 0 or hi < 0)})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "bakeoff.csv", index=False)
    (REPORTS / "bakeoff.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    show = ["model", "roc_auc", "pr_auc", "P@2pct", "P@5pct", "P@2pct_diff_vs_lightgbm", "diff_ci_low", "diff_ci_high"]
    print(out[[c for c in show if c in out]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
