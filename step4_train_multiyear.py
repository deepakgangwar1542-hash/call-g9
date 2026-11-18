"""Multi-year, leak-free temporal evaluation and training for call triage (v2).

Protocol (every split is by calendar year, never random):
  selection : train 2021..Y-2  -> validate Y-1   (choose feature tier, set nothing on the test year)
  final test: train 2021..Y-1  -> test Y         (Y = latest year in the data, untouched until the end)
  backtest  : for every year Y with >= 2 earlier years, train on all earlier years and test on Y
Baseline is the Level 1 speed-limit rate. Ranking metrics (P@K%) are used because the product escalates a fixed share.
"""
import argparse
import datetime as dt
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common import (FLAG_RATE, K_DEFAULT, LGBM_PARAMS, MULTI_CATEGORICAL, MULTI_CLEAN, ROOT, SEED, TIERS,
                    V2_META_PATH, V2_MODEL_PATH, precision_at_k)

REPORTS = ROOT / "reports"
TIE_SEEDS = tuple(SEED + i for i in range(5))
BOOTSTRAPS = 300


def load_clean():
    df = pd.read_csv(MULTI_CLEAN)
    categories = {}
    for col in MULTI_CATEGORICAL:
        values = sorted(df[col].unique(), key=str)
        categories[col] = values
    return df, categories


def to_frame(df, features, categories):
    X = df[features].copy()
    for col in features:
        if col in categories:
            X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def fit_lgbm(train, features, categories):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(to_frame(train, features, categories), train["target"])
    return model


def score_lgbm(model, test, features, categories):
    return model.predict_proba(to_frame(test, features, categories))[:, 1]


def level1_scores(train, test):
    rates = train.groupby("speed_limit")["target"].mean()
    return test["speed_limit"].map(rates).fillna(train["target"].mean()).to_numpy(float)


def p_at_rate(y, scores, rate, seeds=TIE_SEEDS):
    k = max(1, int(np.ceil(rate * len(y))))
    return float(np.mean([precision_at_k(y, scores, k, seed=s) for s in seeds]))


def metrics(y, scores):
    y = np.asarray(y)
    base = float(y.mean())
    p2, p5 = p_at_rate(y, scores, FLAG_RATE), p_at_rate(y, scores, 0.05)
    return {"n": len(y), "base_rate": base, "roc_auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)), "P@2pct": p2, "P@2pct_lift": p2 / base,
            "P@5pct": p5, "P@5pct_lift": p5 / base, "P@300": p_at_rate(y, scores, K_DEFAULT / len(y))}


def bootstrap_vs_baseline(y, model_scores, base_scores, rate=FLAG_RATE, n=BOOTSTRAPS):
    rng, y = np.random.default_rng(SEED), np.asarray(y)
    own, diff = [], []
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        k = max(1, int(np.ceil(rate * len(idx))))
        m = precision_at_k(y[idx], model_scores[idx], k, seed=SEED + i)
        b = precision_at_k(y[idx], base_scores[idx], k, seed=SEED + i)
        own.append(m); diff.append(m - b)
    lo, hi = np.percentile(own, [2.5, 97.5]); dlo, dhi = np.percentile(diff, [2.5, 97.5])
    return {"P@2pct_ci_low": float(lo), "P@2pct_ci_high": float(hi), "diff_vs_level1_mean": float(np.mean(diff)),
            "diff_vs_level1_ci_low": float(dlo), "diff_vs_level1_ci_high": float(dhi),
            "ci_excludes_zero": bool(dlo > 0 or dhi < 0)}


def evaluate(df, categories, train_years, test_year, tiers=TIERS):
    train, test = df[df.year.isin(train_years)], df[df.year == test_year]
    rows, scores = [], {"Level1_speed_limit_rate": level1_scores(train, test)}
    rows.append({"tier": "Level1_speed_limit_rate", "n_features": 1, **metrics(test.target, scores["Level1_speed_limit_rate"])})
    models = {}
    for tier, features in tiers.items():
        models[tier] = fit_lgbm(train, features, categories)
        scores[tier] = score_lgbm(models[tier], test, features, categories)
        rows.append({"tier": tier, "n_features": len(features), **metrics(test.target, scores[tier])})
    out = pd.DataFrame(rows)
    out.insert(0, "test_year", test_year)
    out.insert(1, "train_years", f"{min(train_years)}-{max(train_years)}")
    return out, scores, models, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="auto", choices=["auto", *TIERS],
                        help="feature tier to deploy; auto = best validation P@2%% (see caveat on police_force)")
    args = parser.parse_args()
    if not MULTI_CLEAN.exists():
        raise SystemExit("Run step0_download.py and step2b_clean_multi.py first.")
    REPORTS.mkdir(exist_ok=True)
    df, categories = load_clean()
    years = sorted(df.year.unique())
    test_year, val_year = years[-1], years[-2]
    print(f"Years: {years} | validation={val_year} | final test={test_year}")

    # 1) Selection: train < val_year, validate on val_year
    sel, sel_scores, _, val = evaluate(df, categories, [y for y in years if y < val_year], val_year)
    sel["split"] = "validation"
    best = sel[sel.tier.isin(TIERS)].sort_values(["P@2pct", "pr_auc"], ascending=False).iloc[0]["tier"]
    chosen = best if args.tier == "auto" else args.tier
    print(f"Validation winner: {best} | deploying: {chosen}")

    # 2) Final test: train everything before the test year, evaluate once
    fin, fin_scores, fin_models, test = evaluate(df, categories, [y for y in years if y < test_year], test_year)
    fin["split"] = "final_test"
    ci = {t: bootstrap_vs_baseline(test.target, fin_scores[t], fin_scores["Level1_speed_limit_rate"]) for t in TIERS}
    fin = fin.merge(pd.DataFrame(ci).T.rename_axis("tier").reset_index(), on="tier", how="left")

    # 3) Rolling-origin backtest: stability across every year with >= 2 earlier years
    back = pd.concat([evaluate(df, categories, [y for y in years if y < t], t)[0] for t in years[2:]], ignore_index=True)

    comparison = pd.concat([sel, fin], ignore_index=True)
    comparison.to_csv(REPORTS / "multiyear_comparison.csv", index=False)
    back.to_csv(REPORTS / "multiyear_backtest.csv", index=False)
    drift = df.groupby("year").agg(rows=("target", "size"), serious_fatal_share=("target", "mean")).reset_index()
    drift.to_csv(REPORTS / "multiyear_base_rates.csv", index=False)

    # Calibration of the chosen tier on the final test (Brier vs the constant base-rate forecast)
    p = fin_scores[chosen]
    brier, brier_const = brier_score_loss(test.target, p), brier_score_loss(test.target, np.full(len(test), df[df.year < test_year].target.mean()))

    # 4) Operating threshold: out-of-time scores of the pre-test model on the test year, top FLAG_RATE share
    threshold = float(np.quantile(p, 1 - FLAG_RATE))
    actual_flag = float((fin_models[chosen].predict_proba(to_frame(test, TIERS[chosen], categories))[:, 1] >= threshold).mean())

    # 5) Deployment model: refit on ALL years with the chosen tier
    deploy = fit_lgbm(df, TIERS[chosen], categories)
    row = fin[(fin.tier == chosen)].iloc[0]
    meta = {"model_version": f"LightGBM-{chosen}-{years[0]}_{years[-1]}-{dt.datetime.now():%Y%m%d%H%M}",
            "tier": chosen, "features": TIERS[chosen], "train_years": [int(y) for y in years], "threshold": threshold,
            "flag_rate": FLAG_RATE, "base_rate": float(df.target.mean()), "test_year": int(test_year),
            "metrics_final_test": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items() if k not in ("tier", "split", "train_years")},
            "brier": float(brier), "brier_constant_forecast": float(brier_const),
            "eval_protocol": "temporal: select on validation year, single final test on latest year, rolling backtest"}
    joblib.dump({"model": deploy, "features": TIERS[chosen], "categories": {c: categories[c] for c in TIERS[chosen] if c in categories}, "tier": chosen}, V2_MODEL_PATH)
    V2_META_PATH.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    show = ["tier", "n_features", "roc_auc", "P@2pct", "P@2pct_lift", "P@5pct"]
    print(f"\nVALIDATION ({val_year}, train <{val_year})\n", sel[show].round(3).to_string(index=False))
    print(f"\nFINAL TEST ({test_year}, train <{test_year})\n", fin[show + ["P@2pct_ci_low", "P@2pct_ci_high", "diff_vs_level1_ci_low", "diff_vs_level1_ci_high"]].round(3).to_string(index=False))
    print("\nROLLING BACKTEST (P@2%)\n", back.pivot(index="tier", columns="test_year", values="P@2pct").round(3).to_string())
    print("\nBASE RATES\n", drift.round(3).to_string(index=False))
    print(f"\nBrier {brier:.4f} vs constant {brier_const:.4f} | threshold {threshold:.4f} | actual flag rate {actual_flag:.2%}")
    print("Saved:", V2_MODEL_PATH.name, V2_META_PATH.name)


if __name__ == "__main__":
    main()
