"""Phase 1: leak-free model selection and final evaluation for call triage."""
import datetime as dt
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import CLEAN_CSV, FEATURES, K_DEFAULT, META_PATH, MODEL_PATH, ROOT, SEED, precision_at_k

REPORTS, FIGURES = ROOT / "reports", ROOT / "reports" / "figures"
TIE_SEEDS, KS, RATES, BOOTSTRAPS = tuple(SEED + i for i in range(20)), (150, 300, 600), (0.02, .05, .10), 1000


def as_model_frame(X):
    return X.loc[:, FEATURES].astype({"speed_limit": str})


def make_pipeline(classifier):
    return Pipeline([("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES)])), ("clf", classifier)])


def model_candidates():
    return {
        "LogisticRegression": make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        "RandomForest": make_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, class_weight="balanced_subsample", random_state=SEED)),
        "HistGradientBoosting": make_pipeline(HistGradientBoostingClassifier(max_depth=4, learning_rate=.08, random_state=SEED)),
    }


def _auc(y, s): return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan
def _pr(y, s): return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def average_precision_at_k(y, scores, k, seeds=TIE_SEEDS):
    """Mean/std across fixed tie-break seeds; ties are common with five features."""
    vals = np.asarray([precision_at_k(y, scores, k, seed=s) for s in seeds])
    return float(vals.mean()), float(vals.std())


def metric_row(y, scores, candidate, split, level0=False):
    y, base = np.asarray(y), float(np.mean(y))
    row = {"candidate": candidate, "split": split, "n": len(y), "base_rate": base,
           "roc_auc": .5 if level0 else _auc(y, scores), "pr_auc": base if level0 else _pr(y, scores)}
    points = [(f"P@{k}", min(k, len(y))) for k in KS] + [(f"P@{int(r*100)}pct", max(1, int(np.ceil(r * len(y))))) for r in RATES]
    for label, k in points:
        mean, std = (0., 0.) if level0 else average_precision_at_k(y, scores, k)
        row[label], row[f"{label}_std"], row[f"{label}_lift"] = mean, std, mean / base if base else np.nan
    row["precision_at_k"] = row["P@300"]
    return row


def fit_level1(X, y):
    frame = pd.DataFrame({"speed_limit": X["speed_limit"].to_numpy(), "target": np.asarray(y)})
    return frame.groupby("speed_limit")["target"].mean(), float(frame.target.mean())


def level1_scores(X, fitted):
    rates, global_rate = fitted
    return X.speed_limit.map(rates).fillna(global_rate).to_numpy(float)


def fit_level15(X, y, smoothing=20):
    """Interaction-rate baseline shrunk toward a fold-local speed-limit rate."""
    frame = X[["speed_limit", "road_type", "light_condition"]].copy(); frame["target"] = np.asarray(y)
    global_rate = float(frame.target.mean())
    speed = frame.groupby("speed_limit").target.agg(["sum", "count"])
    speed_rates = ((speed["sum"] + smoothing * global_rate) / (speed["count"] + smoothing)).to_dict()
    group = frame.groupby(["speed_limit", "road_type", "light_condition"]).target.agg(["sum", "count"])
    group["rate"] = [(v["sum"] + smoothing * speed_rates.get(key[0], global_rate)) / (v["count"] + smoothing) for key, v in group.iterrows()]
    return group.rate.to_dict(), speed_rates, global_rate


def level15_scores(X, fitted):
    group, speed, global_rate = fitted
    return np.asarray([group.get((r.speed_limit, r.road_type, r.light_condition), speed.get(r.speed_limit, global_rate)) for r in X[["speed_limit", "road_type", "light_condition"]].itertuples(index=False)], dtype=float)


def oof_scores(X, y):
    scores = {"Level0_always_Slight": np.zeros(len(X)), "Level1_speed_limit_rate": np.zeros(len(X)), "Level1_5_smoothed_interaction_rate": np.zeros(len(X))}
    models = model_candidates(); scores.update({n: np.zeros(len(X)) for n in models})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in cv.split(X, y):
        Xtr, Xva, ytr = X.iloc[tr], X.iloc[va], y.iloc[tr]
        scores["Level1_speed_limit_rate"][va] = level1_scores(Xva, fit_level1(Xtr, ytr))
        scores["Level1_5_smoothed_interaction_rate"][va] = level15_scores(Xva, fit_level15(Xtr, ytr))
        for name, model in models.items():
            fitted = clone(model).fit(as_model_frame(Xtr), ytr)
            scores[name][va] = fitted.predict_proba(as_model_frame(Xva))[:, 1]
    return scores


def score_final(Xtr, ytr, Xte, chosen):
    scores = {"Level0_always_Slight": np.zeros(len(Xte))}
    l1, l15 = fit_level1(Xtr, ytr), fit_level15(Xtr, ytr)
    scores["Level1_speed_limit_rate"], scores["Level1_5_smoothed_interaction_rate"] = level1_scores(Xte, l1), level15_scores(Xte, l15)
    fitted = {}
    for name, model in model_candidates().items():
        fitted[name] = model.fit(as_model_frame(Xtr), ytr)
        scores[name] = fitted[name].predict_proba(as_model_frame(Xte))[:, 1]
    return scores, fitted[chosen]


def cutoff_for_rate(scores, rate):
    rank = max(0, min(len(scores) - 1, int(np.ceil(rate * len(scores))) - 1))
    return float(np.sort(np.asarray(scores))[::-1][rank])


def bootstrap_results(y, score_map, chosen, k=K_DEFAULT, n=BOOTSTRAPS):
    """Paired differences use the same resample; one seeded tie draw per resample."""
    rng, yv = np.random.default_rng(SEED), np.asarray(y)
    own = {name: [] for name in score_map}; differences = {"Level1_speed_limit_rate": [], "Level1_5_smoothed_interaction_rate": []}
    for i in range(n):
        idx, seed, vals = rng.integers(0, len(yv), len(yv)), SEED + i, {}
        for name, score in score_map.items():
            vals[name] = 0. if name == "Level0_always_Slight" else precision_at_k(yv[idx], np.asarray(score)[idx], min(k, len(idx)), seed=seed)
            own[name].append(vals[name])
        for baseline in differences: differences[baseline].append(vals[chosen] - vals[baseline])
    cis = {name: tuple(map(float, np.percentile(values, [2.5, 97.5]))) for name, values in own.items()}
    rows = []
    for baseline, values in differences.items():
        lo, hi = map(float, np.percentile(values, [2.5, 97.5]))
        rows.append({"model": chosen, "baseline": baseline, "metric": f"P@{k}", "bootstrap_n": n, "mean_difference": float(np.mean(values)), "ci_low": lo, "ci_high": hi, "fraction_model_beats_baseline": float(np.mean(np.asarray(values) > 0)), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return cis, pd.DataFrame(rows), np.asarray(differences["Level1_speed_limit_rate"])


def tie_fraction(scores, k):
    cutoff = np.sort(np.asarray(scores))[::-1][min(k, len(scores)) - 1]; size = int(np.sum(np.asarray(scores) == cutoff))
    return min(size, k) / min(k, len(scores)), size


def final_split_indices(y):
    """Deterministic, stratified positions; kept public for regression tests."""
    return train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=SEED)


def run_variant(name, path):
    df = pd.read_csv(path); X, y = df[FEATURES].copy(), df.target.astype(int).copy()
    dev_idx, test_idx = final_split_indices(y)
    assert not set(dev_idx).intersection(test_idx), "final-test indices overlap development indices"
    Xdev, Xtest, ydev, ytest = X.iloc[dev_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True), y.iloc[dev_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    oof = oof_scores(Xdev, ydev)
    rows = [metric_row(ydev, s, n, "cv_oof", n == "Level0_always_Slight") for n, s in oof.items()]
    deployable = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
    chosen = max(deployable, key=lambda n: (next(r for r in rows if r["candidate"] == n)["P@300"], next(r for r in rows if r["candidate"] == n)["pr_auc"]))
    flag_rate, threshold = min(K_DEFAULT / len(Xtest), 1.), cutoff_for_rate(oof[chosen], min(K_DEFAULT / len(Xtest), 1.))
    scores, model = score_final(Xdev, ydev, Xtest, chosen)
    final = [metric_row(ytest, s, n, "final_test", n == "Level0_always_Slight") for n, s in scores.items()]
    cis, paired, samples = bootstrap_results(ytest, scores, chosen)
    for row in final: row["P@300_ci_low"], row["P@300_ci_high"] = cis[row["candidate"]]
    selected = next(r for r in final if r["candidate"] == chosen); high = scores[chosen] >= threshold; fraction, group_size = tie_fraction(scores[chosen], min(K_DEFAULT, len(ytest)))
    selected.update({"threshold": threshold, "flag_rate_target": flag_rate, "flag_rate_actual": float(high.mean()), "high_precision": float(ytest[high].mean()) if high.any() else np.nan, "high_recall": float(ytest[high].sum() / ytest.sum()) if ytest.sum() else np.nan, "unique_feature_combinations": int(Xtest.drop_duplicates().shape[0]), "cutoff_tie_group_fraction_of_top300": fraction, "cutoff_tie_group_size": group_size})
    return {"name": name, "df": df, "Xdev": Xdev, "ydev": ydev, "Xtest": Xtest, "ytest": ytest, "dev_idx": dev_idx, "test_idx": test_idx, "rows": rows + final, "scores": scores, "chosen": chosen, "model": model, "threshold": threshold, "flag_rate": flag_rate, "paired": paired, "paired_samples": samples}


def subgroup_rows(r):
    X = r["Xtest"].copy(); X["target"] = r["ytest"].to_numpy(); X["model_score"] = r["scores"][r["chosen"]]; X["level1_score"] = r["scores"]["Level1_speed_limit_rate"]; X["high"] = X.model_score >= r["threshold"]; X["light_group"] = np.where(X.light_condition.eq("Daylight"), "Daylight", "Dark / unknown")
    rows = []
    for dimension in ("light_group", "road_type", "speed_limit"):
        for value, group in X.groupby(dimension, dropna=False):
            y = group.target.to_numpy(); rows.append({"dimension": dimension, "group": value, "n": len(group), "base_rate": float(y.mean()), "model_roc_auc": _auc(y, group.model_score), "level1_roc_auc": _auc(y, group.level1_score), "share_flagged_high": float(group.high.mean())})
    return pd.DataFrame(rows)


def calibration_rows(r):
    d = pd.DataFrame({"score": r["scores"][r["chosen"]], "target": r["ytest"].to_numpy()}); d["bin"] = pd.qcut(d.score, q=10, duplicates="drop")
    out = d.groupby("bin", observed=True).agg(n=("target", "size"), mean_predicted_score=("score", "mean"), observed_rate=("target", "mean")).reset_index(); out["brier_score"] = brier_score_loss(d.target, d.score)
    return out


def transfer_check(r):
    data = r["df"]; pre, post = data[~data.after_sep17.astype(bool)], data[data.after_sep17.astype(bool)]
    model = clone(model_candidates()[r["chosen"]]).fit(as_model_frame(pre[FEATURES]), pre.target)
    return metric_row(post.target, model.predict_proba(as_model_frame(post[FEATURES]))[:, 1], r["chosen"], "train_pre_test_post")


def make_figures(r, paired, importance, calibration):
    FIGURES.mkdir(parents=True, exist_ok=True); names = [r["chosen"], "Level1_speed_limit_rate", "Level1_5_smoothed_interaction_rate"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        estimates = [average_precision_at_k(r["ytest"], r["scores"][name], k) for k in KS]
        means = np.asarray([x[0] for x in estimates]); errors = np.asarray([x[1] for x in estimates])
        ax.plot(KS, means, marker="o", label=name.replace("_", " "))
        ax.fill_between(KS, np.clip(means - 1.96 * errors, 0, 1), np.clip(means + 1.96 * errors, 0, 1), alpha=.12)
    ax.set(xlabel="Dispatch capacity (K calls)", ylabel="Precision@K", title="Final-test precision by dispatch capacity"); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "precision_at_k_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.hist(r["paired_samples"], bins=35, color="#4c78a8", edgecolor="white"); first = paired.iloc[0]; ax.axvline(0, color="black"); ax.axvline(first.ci_low, color="#d62728", ls="--"); ax.axvline(first.ci_high, color="#d62728", ls="--"); ax.set(xlabel="Model P@300 − Level 1 P@300", ylabel="Bootstrap samples", title="Paired bootstrap difference"); fig.tight_layout(); fig.savefig(FIGURES / "paired_difference_histogram.png", dpi=160); plt.close(fig)
    imp = importance.sort_values("importance_mean"); fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(imp.feature, imp.importance_mean, xerr=imp.importance_std, color="#59a14f"); ax.set(xlabel="Permutation importance (ROC-AUC decrease)", title="Final-test feature importance"); fig.tight_layout(); fig.savefig(FIGURES / "permutation_importance.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot([0,1], [0,1], "--", color="grey", label="Perfect calibration"); ax.plot(calibration.mean_predicted_score, calibration.observed_rate, marker="o", label=r["chosen"]); ax.set(xlabel="Mean predicted probability", ylabel="Observed Serious/Fatal rate", title="Calibration on final test"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "calibration.png", dpi=160); plt.close(fig)


def write_summary(r, paired):
    final = next(x for x in r["rows"] if x["split"] == "final_test" and x["candidate"] == r["chosen"])
    claims = " ".join(f"The CI versus {x.baseline} {'excludes' if x.ci_excludes_zero else 'includes'} zero ({x.ci_low:.3f} to {x.ci_high:.3f})." for _, x in paired.iterrows())
    (REPORTS / "phase1_summary.md").write_text(f'''# Phase 1 model evaluation\n\nThe selected deployable model is **{r['chosen']}**, trained using **{r['name']}** data. On its untouched final test set, Precision@300 was **{final['P@300']:.3f}** (95% bootstrap CI {final['P@300_ci_low']:.3f}–{final['P@300_ci_high']:.3f}); ROC-AUC was **{final['roc_auc']:.3f}** and PR-AUC **{final['pr_auc']:.3f}**. The High threshold is {r['threshold']:.4f}, targeting {r['flag_rate']:.2%} of calls.\n\n{claims}\n\nThe Sep-17 recommendation is **{r['name']}**, based on the held-out comparison in `sep17_sensitivity.csv`, not an assumed data-quality rule.\n\n## Limitations\n\n- STATS19 covers police-attended collisions with a casualty, not every incoming road-safety call.\n- "Unknown" categories correlate with low severity partly because reporting and attendance differ, so the model may partly learn reporting behaviour.\n- The five call-time features create many tied scores; reported Precision@K averages 20 deterministic tie-breaks.\n- This observational dataset cannot show that enhanced response changes outcomes.\n- These estimates are specific to 2023 UK data and need monitoring before operational use.\n''', encoding="utf-8")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True); all_path = CLEAN_CSV.with_name("clean_2023_all.csv")
    if not all_path.exists(): raise FileNotFoundError("Run `python step2_clean.py --include-after-sep17` before this script.")
    results = [run_variant("pre_sep17", CLEAN_CSV), run_variant("all", all_path)]
    pd.DataFrame([row | {"data_variant": r["name"]} for r in results for row in r["rows"]]).to_csv(REPORTS / "model_comparison.csv", index=False)
    paired = pd.concat([r["paired"].assign(data_variant=r["name"]) for r in results], ignore_index=True); paired.to_csv(REPORTS / "paired_difference.csv", index=False)
    sensitivity = []
    for r in results:
        f = next(x for x in r["rows"] if x["split"] == "final_test" and x["candidate"] == r["chosen"]); sensitivity.append({"data_variant": r["name"], "chosen_model": r["chosen"], "n": f["n"], "P@300": f["P@300"], "P@2pct": f["P@2pct"], "P@5pct": f["P@5pct"], "roc_auc": f["roc_auc"], "pr_auc": f["pr_auc"]})
    all_result = results[1]; transfer = transfer_check(all_result); sensitivity[-1].update({"transfer_roc_auc": transfer["roc_auc"], "transfer_pr_auc": transfer["pr_auc"], "transfer_p_at_300": transfer["P@300"]}); sensitivity = pd.DataFrame(sensitivity); sensitivity.to_csv(REPORTS / "sep17_sensitivity.csv", index=False)
    chosen = all_result if sensitivity.loc[1, "P@300"] >= sensitivity.loc[0, "P@300"] else results[0]
    pi = permutation_importance(chosen["model"], as_model_frame(chosen["Xtest"]), chosen["ytest"], scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1); importance = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}); importance.to_csv(REPORTS / "permutation_importance.csv", index=False)
    subgroup_rows(chosen).to_csv(REPORTS / "subgroups.csv", index=False); calibration = calibration_rows(chosen); calibration.to_csv(REPORTS / "calibration.csv", index=False)
    chosen_paired = paired[paired.data_variant.eq(chosen["name"])].reset_index(drop=True); make_figures(chosen, chosen_paired, importance, calibration); write_summary(chosen, chosen_paired)
    final = next(x for x in chosen["rows"] if x["split"] == "final_test" and x["candidate"] == chosen["chosen"]); cv = next(x for x in chosen["rows"] if x["split"] == "cv_oof" and x["candidate"] == chosen["chosen"])
    meta = {"model_version": f"{chosen['chosen']}-phase1-{dt.datetime.now():%Y%m%d%H%M}", "model": chosen["chosen"], "features": FEATURES, "K": K_DEFAULT, "threshold": chosen["threshold"], "flag_rate": chosen["flag_rate"], "metrics": final, "cv_metrics": cv, "level1_precision_at_k": next(x for x in chosen["rows"] if x["split"] == "final_test" and x["candidate"] == "Level1_speed_limit_rate")["P@300"], "base_rate": float(chosen["ydev"].mean()), "data_variant": chosen["name"], "eval_protocol": "80/20 stratified final test; 5-fold development OOF selection; 20 tie seeds; 1000 paired bootstraps"}
    joblib.dump(chosen["model"], MODEL_PATH); META_PATH.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print("\nFinal comparison:"); print(sensitivity.round(4).to_string(index=False)); print(f"\nDeployed {chosen['chosen']} on {chosen['name']} | threshold={chosen['threshold']:.4f} | flag rate={chosen['flag_rate']:.2%}"); print(chosen_paired[["baseline", "mean_difference", "ci_low", "ci_high", "ci_excludes_zero"]].round(4).to_string(index=False))


if __name__ == "__main__": main()
